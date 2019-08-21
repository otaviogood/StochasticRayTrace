/*--------------------------------------------------------------------------------------
License CC0 - http://creativecommons.org/publicdomain/zero/1.0/
To the extent possible under law, the author(s) have dedicated all copyright and related and neighboring rights to this software to the public domain worldwide. This software is distributed without any warranty.
----------------------------------------------------------------------------------------
^This means do anything you want with this code. Because we are programmers, not lawyers.

This is the latest iteration of my stochastic ray tracer.
For materials, it supports:
- Proper reflection / refraction, with total internal reflection.
- Filtered light rays like in the red glass.
- Gold has rgb reflectance.
- Emission.
- Diffuse / specular.
- Sub surface scattering, like in water.
- Scattering in air or whereever for fog.

- Support for ray traced and ray marched geometry in the same scene.
- Antialiasing.
- Depth of field

Random number generation and hashes were cleaned up and now feel pretty good.

-Otavio Good
*/

// Number of samples per pixel - bigger takes more compute
#define NUM_SAMPLES 1
// Number of times the ray bounces off things before terminating
#define NUM_ITERS 7

// ---- general helper functions / constants ----
#define saturate(a) clamp(a, 0.0, 1.0)
// Weird for loop trick so compiler doesn't unroll loop
// By making the zero a variable instead of a constant, the compiler can't unroll the loop and
// that speeds up compile times by a lot.
#define ZERO_TRICK max(0, -iFrame)
const int BIG_INT = 2000000000;
const float PI = 3.14159265;
const float farPlane = 64.0;

vec3 RotateX(vec3 v, float rad)
{
	float cos = cos(rad);
	float sin = sin(rad);
	return vec3(v.x, cos * v.y + sin * v.z, -sin * v.y + cos * v.z);
}
vec3 RotateY(vec3 v, float rad)
{
	float cos = cos(rad);
	float sin = sin(rad);
	return vec3(cos * v.x - sin * v.z, v.y, sin * v.x + cos * v.z);
}
vec3 RotateZ(vec3 v, float rad)
{
	float cos = cos(rad);
	float sin = sin(rad);
	return vec3(cos * v.x + sin * v.y, -sin * v.x + cos * v.y, v.z);
}
// Find 2 perpendicular vectors to the input vector.
mat3 MakeBasis(vec3 normal) {
	mat3 result;
	result[0] = normal;
	if (abs(normal.y) > 0.5) {
		result[1] = normalize(cross(normal, vec3(1.0, 0.0, 0.0)));
	}
	else {
		result[1] = normalize(cross(normal, vec3(0.0, 1.0, 0.0)));
	}
	result[2] = normalize(cross(normal, result[1]));
	return result;
}


// ---- Random functions use one 32 bit state var to change things up ----
// This is the single state variable for the random number generator.
uint randomState = 4056649889u;
// 0xffffff is biggest 2^n-1 that 32 bit float does exactly.
// Check with Math.fround(0xffffff) in javascript.
const float invMax24Bit = 1.0 / float(0xffffff);

// This is the main hash function that should produce a non-repeating
// pseudo-random sequence for 2^31 iterations.
uint SmallHashA(in uint seed) {
	return (seed ^ 1057926937u) * 3812423987u ^
		((seed*seed) * 4000000007u);
}
// This is an extra hash function to clean things up a little.
uint SmallHashB(in uint seed) {
	return (seed ^ 2156034509u) * 3699529241u;
}

// Hash the random state to get a random float ranged [0..1]
float RandFloat() {
	randomState = SmallHashA(randomState);
	// Add these 2 lines for extra randomness. And change last line to tempState.
	uint tempState = (randomState << 13) | (randomState >> 19);
	tempState = SmallHashB(tempState);
	return float((tempState >> 8) & 0xffffffu) * invMax24Bit;
}
// Hash the random state to get 2 random floats ranged [0..1]
// Reduced precision to 16 bits per component.
vec2 RandVec2() {
	randomState = SmallHashA(randomState);
	uint tempState = (randomState << 13) | (randomState >> 19);
	tempState = SmallHashB(tempState);
	return vec2(tempState & 0xffffu,
		(tempState >> 16) & 0xffffu) / float(0xffff);
}
// Hash the random state to get 3 random floats ranged [0..1]
// Reduced precision to 10 bits per component.
vec3 RandVec3() {
	randomState = SmallHashA(randomState);
	uint tempState = (randomState << 13) | (randomState >> 19);
	tempState = SmallHashB(tempState);
	return vec3((tempState >> 2) & 0x3ffu,
		(tempState >> 12) & 0x3ffu,
		(tempState >> 22) & 0x3ffu) / float(0x3ffu);
}

// Returns a random float from [0..1]
float HashFloat(uint seed) {
	seed = SmallHashA(seed);
	return float((seed >> 8) & 0xffffffu) * invMax24Bit;
}
vec2 HashVec2(uint seed) {
	seed = SmallHashA(seed);
	seed = (seed << 13) | (seed >> 19);
	seed = SmallHashB(seed);
	return vec2(seed & 0xffffu,
		(seed >> 16) & 0xffffu) / float(0xffff);
}
vec3 HashVec3(uint seed) {
	seed = SmallHashA(seed);
	seed = (seed << 13) | (seed >> 19);
	seed = SmallHashB(seed);
	return vec3((seed >> 2) & 0x3ffu,
		(seed >> 12) & 0x3ffu,
		(seed >> 22) & 0x3ffu) / float(0x3ffu);
}
float HashFloatI2(ivec2 seed) {
	uint seedB = SmallHashA(uint(seed.x ^ (seed.y * 65537)));
	//seedB ^= SmallHashB(uint(seed.y));
	return float(seedB & 0xffffffu) * invMax24Bit;
}

void SetRandomSeed(in vec2 fragCoord, in vec2 iResolution,
	in int iFrame) {
	uint primex = max(uint(iResolution.x), 5003u);  // This prime is far from any 2^x
	randomState = uint(fragCoord.x);
	randomState += uint(fragCoord.y) * primex;
	randomState += uint(iFrame)* primex * uint(iResolution.y);
	RandFloat();
}

// Returns random number sampled from a circular gaussian distribution
// https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
vec2 RandGaussianCircle() {
	vec2 u = RandVec2();
	u.x = max(u.x, 0.00000003); // We don't want log() to fail because it's 0.
	float a = sqrt(-2.0 * log(u.x));
	return vec2(a * cos(2.0*PI*u.y), a * sin(2.0 * PI * u.y));
}
// Randomly sample the lifetime of a ray moving through particles
// p = prob. of collision with particle per unit distance traveled
float RayLifetime(float p) {
	if (p < 0.00000003) return farPlane;  // Lower than this and the math seems to fail.
	float unif = RandFloat();  // uniform(0, 1)
	// This random shouldn't be allowed to hit 0 because log() is undefined.
	unif = max(0.00000000000001, unif);
	// p can't be 0 because log(1) == 0 and it divides by 0. Can't be 1 because log(0) is undefined.
	p = min(p, .999999);
	//float g = math.floor(math.log(unif) / math.log(1 - p))  # natural logs
	// g = number of successes before failure
	float g = log(unif) / log(1.0 - p);
	return g;
}

vec3 HashPointOnSphere(uint seed) {
	vec2 uv = HashVec2(seed);
	float theta = 2.0 * PI * uv.x;
	float psi = acos(2.0 * uv.y - 1.0);
	float x = cos(theta) * sin(psi);
	float y = sin(theta) * sin(psi);
	float z = cos(psi);
	return vec3(x, y, z);
}

// Random point *ON* sphere
vec3 RandPointOnSphere() {
	vec2 uv = RandVec2();
	float theta = 2.0 * PI * uv.x;
	float psi = acos(2.0 * uv.y - 1.0);
	float x = cos(theta) * sin(psi);
	float y = sin(theta) * sin(psi);
	float z = cos(psi);
	return vec3(x, y, z);
}
// Random point *IN* sphere
// This is biased!!! About 1/32 of the time, it will return a point in box instead of a sphere.
/*vec3 RandPointInSphere() {
	return RandPointOnSphere();
	vec3 p = Randf3i1(seed) * 2.0 - 1.0;
	if (length(p) <= 1.0) return p;
	p = Randf3i1(seed) * 2.0 - 1.0;
	if (length(p) <= 1.0) return p;
	p = Randf3i1(seed) * 2.0 - 1.0;
	if (length(p) <= 1.0) return p;
	p = Randf3i1(seed) * 2.0 - 1.0;
	if (length(p) <= 1.0) return p;
	p = Randf3i1(seed) * 2.0 - 1.0;
	if (length(p) <= 1.0) return p;
	return p;
}*/

// ---- Environment maps - a few to choose from ----
// Make a procedural environment map with a giant softbox light and 4 lights around the sides.
vec3 GetEnvMap3(vec3 rayDir)
{
	// fade bottom to top so it looks like the softbox is casting light on a floor
	// and it's bouncing back
	vec3 final = vec3(1.0) * dot(rayDir, vec3(0.0, 1.0, 0.0)) * 0.5 + 0.5;
	final *= 0.125;
	// overhead softbox, stretched to a rectangle
	if ((rayDir.y > abs(rayDir.x)*1.0) && (rayDir.y > abs(rayDir.z*0.25))) final = vec3(2.0)*rayDir.y;
	// fade the softbox at the edges with a rounded rectangle.
	float roundBox = length(max(abs(rayDir.xz / max(0.0, rayDir.y)) - vec2(0.9, 4.0), 0.0)) - 0.1;
	final += vec3(0.8)* pow(saturate(1.0 - roundBox * 0.5), 6.0);
	// purple lights from side
	final += vec3(8.0, 6.0, 7.0) * saturate(0.001 / (1.0 - abs(rayDir.x)));
	// yellow lights from side
	final += vec3(8.0, 7.0, 6.0) * saturate(0.001 / (1.0 - abs(rayDir.z)));
	return vec3(final);
}

// Courtyard environment map texture with extra sky brightness for HDR look.
vec3 GetEnvMap(vec3 rayDir) {
	vec3 tex = texture(iChannel1, rayDir).xyz;
	tex = tex * tex;  // gamma correct
	vec3 light = vec3(0.0);
	// overhead softbox, stretched to a rectangle
	if ((rayDir.y > abs(rayDir.x + 0.6)*0.29) && (rayDir.y > abs(rayDir.z*2.5))) light = vec3(2.0)*rayDir.y;
	vec3 texp = pow(tex, vec3(14.0));
	light *= texp;  // Masked into the existing texture's sky
	return (tex + light * 3.0);
}

vec3 GetEnvMap2(vec3 rayDir) {
	//return vec3(0.0);
	return vec3(rayDir.y*0.5 + 0.5);// * vec3(1.0, 0.5, 0.7);
}

// ---- Ray intersection functions and data structures ----
struct Ray
{
	vec3 p0, dirNormalized;
	int outside;  // 1 ray is outside, -1 ray is inside, 0 terminate ray tracing iteration
};
struct SceneHit
{
	vec3 hitPos;
	vec3 hitNormal;
	float pt;  // parametric t variable - how far along the ray vector is the intersection
	int objIndex;  // unique index per object - used for material tricks like hashing colors
	int materialIndex;  // Which material are we using
};

// As the ray bounces, it records hits in these vars.
struct ColorHit {
	vec3 diffuse;
	vec3 emission;
};
ColorHit colorHits[NUM_ITERS];
int colorHitIndex;
void ResetColorHitList() {
	colorHitIndex = 0;
	for (int i = 0; i < NUM_ITERS; i++) {
		colorHits[i].emission.x = -1.0;
	}
}
void SaveHit(in vec3 diffuse, in vec3 emission) {
	colorHits[colorHitIndex] = ColorHit(diffuse, emission);
	colorHitIndex++;
}

// dirVec MUST BE NORMALIZED FIRST!!!!
float SphereIntersect(vec3 pos, vec3 dirVecPLZNormalizeMeFirst, vec3 spherePos, float rad)
{
	vec3 radialVec = pos - spherePos;
	float b = dot(radialVec, dirVecPLZNormalizeMeFirst);
	float c = dot(radialVec, radialVec) - rad * rad;
	float h = b * b - c;
	if (h < 0.0) return -1.0;
	return -b - sqrt(h);
}

// Return value is normal in xyz, t in w.
// outside is 1 to intersect from the outside of the sphere, -1 to intersect from inside of sphere.
vec4 SphereIntersect3(vec3 pos, vec3 dirVecPLZNormalizeMeFirst, vec3 spherePos, float rad, int outside)
{
	vec4 rh = vec4(farPlane);
	vec3 delta = spherePos - pos;
	float projdist = dot(delta, dirVecPLZNormalizeMeFirst);
	vec3 proj = dirVecPLZNormalizeMeFirst * projdist;
	vec3 bv = proj - delta;
	float b2 = dot(bv, bv);
	if (b2 > rad*rad) return rh;  // Ray missed the sphere
	float x = sqrt(rad*rad - b2);
	rh.w = projdist - (x * float(outside));
	vec3 hitPos = pos + dirVecPLZNormalizeMeFirst * rh.w;
	rh.xyz = normalize(hitPos - spherePos);//*float(outside);//??HACK>>>??  // Normal still points outwards if collision from inside.
	return rh;
}

// https://tavianator.com/fast-branchless-raybounding-box-intersections/
// Return value is normal in xyz, t in w.
// **** rayInv is 1.0 / direction vector ****
vec4 BoxIntersect(vec3 pos, vec3 rayInv, vec3 boxPos, vec3 rad, int outside)
{
	vec3 bmin = boxPos - rad;
	vec3 bmax = boxPos + rad;
	//    vec3 rayInv = 1.0 / dirVecPLZNormalizeMeFirst;

	vec3 t1 = (bmin - pos) * rayInv;
	vec3 t2 = (bmax - pos) * rayInv;

	vec3 vmin = min(t1, t2);
	vec3 vmax = max(t1, t2);

	float tNear = max(vmin.z, max(vmin.x, vmin.y));
	float tFar = min(vmax.z, min(vmax.x, vmax.y));

	vec4 rh = vec4(0, 1, 0, farPlane);
	if ((tFar < tNear)) return rh;
	if (outside > 0) {
		if ((tNear <= 0.0)) return rh;
		rh.w = tNear;
	}
	else if (outside < 0) {
		if ((tFar <= 0.0)) return rh;
		rh.w = tFar;
	}

	// optimize me!
	if (t1.x == rh.w) rh.xyz = vec3(-1.0, 0.0, 0.0);
	else if (t2.x == rh.w) rh.xyz = vec3(1.0, 0.0, 0.0);
	else if (t1.y == rh.w) rh.xyz = vec3(0.0, -1.0, 0.0);
	else if (t2.y == rh.w) rh.xyz = vec3(0.0, 1.0, 0.0);
	else if (t1.z == rh.w) rh.xyz = vec3(0.0, 0.0, -1.0);
	else if (t2.z == rh.w) rh.xyz = vec3(0.0, 0.0, 1.0);
	//	rh.xyz = rh.xyz * float(outside);  // Use this for normal to point inside if hit from inside
	return rh;
}

float iPlane(in vec3 ro, in vec3 rd, in vec2 distBound, inout vec3 normal,
	in vec3 planeNormal, in float planeDist) {
	float a = dot(rd, planeNormal);
	float d = -(dot(ro, planeNormal) + planeDist) / a;
	if (a > 0. || d < distBound.x || d > distBound.y) {
		return farPlane;
	}
	else {
		normal = planeNormal;
		return d;
	}
}


// ---- Scattering (fog) vars ----
// Scattering needs a stack because it's all about what you're moving through.
// So if you move through fog and then through glass, when you come out of the glass,
// you're back into the fog.

// This is the scatter var for the outer-most place. RGB fog diffuse, fog density.
// **** Set fog here ****
const vec4 globalScatter = vec4(0.3, 0.4, 0.6, 0.5);//.5
// Scatter stack
vec4 scatterStack[NUM_ITERS * 2];  // Size correct?
int scatterStackIndex;
vec4 PeekScatter() {
	return scatterStack[scatterStackIndex - 1];
}
void PushScatter(in vec4 s) {
	if (s != PeekScatter())
	{
		scatterStack[scatterStackIndex] = s;
		scatterStackIndex++;
	}
}
vec4 PopScatter(in vec4 s) {
	if (s != PeekScatter())
	{
		scatterStackIndex--;
		return scatterStack[scatterStackIndex];
	}
}
void InitScatterStack(in vec4 s) {
	scatterStack[0] = s;
	scatterStackIndex = 1;
}

// ---- Materials ----

// List of refraction values for different materials
// Linear reflectance values from http://blog.selfshadow.com/publications/s2012-shading-course/hoffman/s2012_pbs_physics_math_notes.pdf
// In w, refrective index - https://en.wikipedia.org/wiki/List_of_refractive_indices
// These can be calculated from each other: https://en.wikipedia.org/wiki/Schlick%27s_approximation
const vec4 refDebug = vec4(1.0, 1.0, 1.0, 1.005);
const vec4 refNone = vec4(0.0, 0.0, 0.0, 1.0);
const vec4 refWater = vec4(0.02, 0.02, 0.02, 1.330);
const vec4 refPlasticGlassLow = vec4(0.03, 0.03, 0.03, 1.42);
const vec4 refGlassWindow = vec4(0.043, 0.043, 0.043, 1.52);
const vec4 refPlasticHigh = vec4(0.05, 0.05, 0.05, 1.58);
const vec4 refGlassHigh = vec4(0.08, 0.08, 0.08, 1.79);  // Ruby
const vec4 refDiamond = vec4(0.172, 0.172, 0.172, 2.417);
// Metals - refractive index is placeholder and shouldn't be used I guess
const vec4 refIron = vec4(0.56, 0.57, 0.58, 1.0);
const vec4 refCopper = vec4(0.95, 0.64, 0.54, 1.0);
const vec4 refGold = vec4(1.0, 0.71, 0.29, 1.0);
const vec4 refAluminum = vec4(0.91, 0.92, 0.92, 1.0);
const vec4 refSilver = vec4(0.95, 0.93, 0.88, 1.0);
const vec4 refSimpleMetal = vec4(0.92, 0.92, 0.92, 1.0);  // rgb is same for better performance

struct Material {
	vec4 reflectRefract;
	float refMult;
	bool doRefraction;
	vec3 diffuse;
	vec3 emission;
	vec3 filterDiffuseDensity;  // This is like a cheap scatter. Not sure I like it.
	vec4 scatter;  // Diffuse in xyz, scatter probability in w.
};
const int defaultMaterialIndex = 3;
const int NUM_MATERIALS = 7;
Material materials[NUM_MATERIALS] = Material[NUM_MATERIALS](
	// water
	Material(refWater, 1.0, true, vec3(1.0), vec3(0.0), vec3(1.0), vec4(0.1, 0.7, 0.99, 0.0)),
	// ruby
	Material(refNone, 0.0, true, vec3(1.0), vec3(0.0), vec3(1.0), vec4(0.995, 0.1, 0.991, 0.5)),
	// gold
	Material(refGold, 1.0, false, vec3(0.0), vec3(0.0), vec3(0.0), vec4(0.0)),
	// colored plastic
	Material(refPlasticGlassLow, 1.0, false, vec3(0.15, 0.15, 0.15), vec3(0.0), vec3(0.0), vec4(0.0)),
	// non-shiny whatever
	Material(refNone, 1.0, false, vec3(0.5), vec3(0.0), vec3(1.0), vec4(0.0)),
	// light
	Material(refNone, 0.0, true, vec3(1.0), vec3(14.0, 12.0, 7.0)*16.0, vec3(0.02), vec4(0.0)),
	// mirrored aluminum
	Material(refNone, 0.0, false, vec3(1.0), vec3(200.0, 250.0, 300.0), vec3(0.0), vec4(0.0))
	);

// ---- Intersect the ray with the scene, ray-trace style ----
SceneHit SceneIntersect(const in Ray ray) {
	SceneHit result;
	result.hitNormal = vec3(0.0);
	result.pt = farPlane;
	result.objIndex = BIG_INT;
	result.materialIndex = defaultMaterialIndex;

	/*SceneHit poly;
	poly.pt = 0.0;
	for (int i = ZERO_TRICK; i <18; i++) {
		vec3 normal = vec3(0,1,0);
		float t = iPlane(ray.p0, ray.dirNormalized, vec2(-200.0, 200.0), normal,
				  HashPointOnSphere(uint(i)), -1.8);
		if ((t > 0.0) && (t < farPlane) && (t > poly.pt)) {
			poly.objIndex = 12399+i;
			poly.materialIndex = 3;// (i&1)*4+1;
			//if ((i & 7) == 0) result.objIndex =4 + ((i/8)&1);
			poly.pt = t;
			poly.hitPos = ray.p0 + ray.dirNormalized * t;
			poly.hitNormal = normal;
		}
	}
	if ((poly.pt < farPlane) && (poly.pt > 0.0)) result = poly;*/

	vec3 rayInv = 1.0 / ray.dirNormalized;
	vec4 sA = vec4(0.0, 0.0, 0.75, 1.5);
	for (int i = ZERO_TRICK; i < 1; i++) {
		sA.xyz = (HashVec3(uint(i + 18)) - 0.5) * 0.0;
		sA.y *= 0.5;
		//sA.y += 0.5;
		//sA.xyz += Randf1i1(uint(float(i)+iTime*60.0)) * sA.xyz * 0.93;
		sA.w = 1.6;// (Hashf1(uint(i+23)) + 0.75) * 1.6;
		float tall = 1.25;
		//if ((i & 7) == 0) {
		//    sA.y += 2.0;
		//    sA.w *= 0.152;
		//    tall = 8.0*Hashf1i1(uint(i+23));
		//}
		vec4 rh = SphereIntersect3(ray.p0, ray.dirNormalized, sA.xyz, sA.w, ray.outside);
		//vec4 rh = BoxIntersect(ray.p0, rayInv, sA.xyz, vec3(sA.w, sA.w*tall, sA.w), ray.outside);
		if ((rh.w > 0.0) && (rh.w < result.pt)) {
			result.objIndex = i + 27;//3;// i&3;
			result.materialIndex = 0;//i+1;//(i&1)*4+1;
			//if ((i & 7) == 0) result.objIndex =4 + ((i/8)&1);
			result.pt = rh.w;
			result.hitPos = ray.p0 + ray.dirNormalized * rh.w;
			result.hitNormal = rh.xyz;
		}
	}
	vec4 rh = BoxIntersect(ray.p0, rayInv, vec3(0, -4.5, 0), vec3(1.0, 0.1, 1.0), ray.outside);
	if ((rh.w > 0.0) && (rh.w < result.pt)) {
		result.objIndex = 1234;
		result.materialIndex = 3;// (i&1)*4+1;
		//if ((i & 7) == 0) result.objIndex =4 + ((i/8)&1);
		result.pt = rh.w;
		result.hitPos = ray.p0 + ray.dirNormalized * rh.w;
		result.hitNormal = rh.xyz;
	}
	rh = BoxIntersect(ray.p0, rayInv, vec3(0.0, 1.95, 0.5), vec3(40.0, 0.05, 0.91), ray.outside);
	if ((rh.w > 0.0) && (rh.w < result.pt)) {
		result.objIndex = 1235;
		result.materialIndex = 5;// (i&1)*4+1;
		//if ((i & 7) == 0) result.objIndex =4 + ((i/8)&1);
		result.pt = rh.w;
		result.hitPos = ray.p0 + ray.dirNormalized * rh.w;
		result.hitNormal = rh.xyz;
	}
	//    IntersectBoxes(vec4(0.0,0.0,0.0,1.0), ray.p0, ray.dirNormalized, ray.outside, result, 0);

	return result;
}

// ---- Also support ray marching, not just ray tracing. ----
// k should be negative. -4.0 works nicely.
// smooth blending function
float smin(float a, float b, float k)
{
	return log2(exp2(k*a) + exp2(k*b)) / k;
}
// min function that supports materials in the y component
vec2 matmin(vec2 a, vec2 b)
{
	if (a.x < b.x) return a;
	else return b;
}
vec2 matmax(vec2 a, vec2 b)
{
	if (a.x > b.x) return a;
	else return b;
}
// signed box distance field
float sdBox(vec3 p, vec3 radius)
{
	vec3 dist = abs(p) - radius;
	return min(max(dist.x, max(dist.y, dist.z)), 0.0) + length(max(dist, 0.0));
}

float sdTorusFlat(vec3 p, vec2 radiuses, float height) {
	vec2 dists = length(p.xz) - radiuses.xy;
	float chop = abs(p.y) - height;
	return max(max(-dists.x, dists.y), chop);
}

// Noise generator from https://otaviogood.github.io/noisegen/
// Params: 3D, Seed 1, Waves 8, Octaves 3, Smooth 1.25
float NoiseGen(vec3 p) {
	// This is a bit faster if we use 2 accumulators instead of 1.
	// Timed on Linux/Chrome/TitanX Pascal
	float wave0 = 0.0;
	float wave1 = 0.0;
	wave0 += sin(dot(p, vec3(-1.593, -1.611, -0.843))) * 0.2286131684;
	wave1 += sin(dot(p, vec3(2.126, -0.284, -1.494))) * 0.1944052014;
	wave0 += sin(dot(p, vec3(3.134, -0.144, 0.623))) * 0.1320359974;
	wave1 += sin(dot(p, vec3(3.291, 2.846, -2.100))) * 0.0659524099;
	wave0 += sin(dot(p, vec3(0.346, 1.573, 4.597))) * 0.0651084310;
	wave1 += sin(dot(p, vec3(-4.428, 0.496, 2.086))) * 0.0641020940;
	wave0 += sin(dot(p, vec3(4.249, -1.624, -3.806))) * 0.0481154253;
	wave1 += sin(dot(p, vec3(-5.350, -4.530, 0.998))) * 0.0370255635;
	return wave0 + wave1;
}
float Apollonian(vec3 pos)
{
	float lenp = length(pos);
	float scale = 1.0;

	//orb = vec4(1000000.0); // max float
	float iter = 0.0;
	//p = clamp(vec3(-1.0,-1.0,-1.0), vec3(1.0,1.0,1.0), p);
	for (int i = 0; i < 10;i++)
	{
		// repeat -1 to 1 region, still scaled to -1 to 1
		pos = fract(0.5*pos + 0.5)*2.0 - 1.0;

		// distance squared from center of 1 repetition region
		float r2 = dot(pos, pos);
		float r3 = pow(abs(pos.x) + abs(pos.y) + abs(pos.z), 1.17);
		r3 *= r3;
		r2 = r3;//(r2+r3+r3)*0.3333;

		// Divide by distance squared
		//float k = max(1.0/r2,0.1);
		// hotspot in middle of region
		float k = (1.45) / r2;
		pos *= k;
		// p = RotateZ(p, iter*sin(iTime)*0.01);
		 //p = RotateY(p, iter*cos(iTime)*PI*1.0);
		scale *= k;
		iter += 1.0;
	}

	return 0.125*abs(pos.y) / scale;
}

// This is the distance function that defines the ray marched scene's geometry.
// The input is a position in space.
// outside is 1 if the ray is intersecting with the outside of objects, -1 for insides (backface)
// The output is the distance to the nearest surface, and a material index
vec2 DistanceToObject(vec3 p, int outside)
{
	vec2 distmat = vec2(Apollonian(p), 4.0);
	vec3 p3 = fract(0.5*p + 0.5)*2.0 - 1.0;
	float d3 = length(p3.xz) - 0.055;
	distmat = matmin(distmat, vec2(d3, 6.0));
	return distmat;
	//float dist = p.y;
	//dist = length(p) - 1.4;
	//dist = smin(dist, length(p + vec3(2.25, -4.0, -4.0)) - 2.95, -0.95);
	float dist = 1000000.0;
	float noise = NoiseGen(p.xyz*2.0)*0.1;

	vec2 water = vec2(dist, 5.0);
	//water = matmin(water, vec2(length(p.xz + vec2(7.0, -5.0))-0.5, 3.0));
	float cyl = length(vec2(p.y, abs(p.x)) + vec2(-5.5, -6.0)) - 0.5;
	cyl = max(cyl, abs(p.z) - 18.0);
	//water = matmin(water, vec2(cyl, 6.0));
	float rad = length(p.xz)*4.0;
	// Make water radial waves (computationally expensive)
	//noise = noise*0.4 -sin(rad)/rad;
	float waterBox = sdBox(p + vec3(0.0, 5.1 + noise, 0.0), vec3(6.0, 0.5, 6.5));
	waterBox = smin(waterBox, length(p + vec3(0.0, 2.95, 0.0)) - 0.6, -3.95);
	//    water = matmin(water, vec2(waterBox, 0.0));
	float pool = sdBox(p + vec3(3.0, 5., 0.0), vec3(3.0, 0.5, 6.5));
	//pool = max(pool, -(length(p.xz) - 4.0));
	//water = matmin(water, vec2(pool, 5.0));
	//water = matmin(water, vec2(sdBox(p + vec3(1.0, -6.1, 0.0), vec3(0.6, 0.1, 8.04)), 6.0));
	//water = matmin(water, vec2(sdBox(p + vec3(1.0, -6.15, 0.0), vec3(0.8, 0.1, 8.4)), 3.0));

	float room = -sdBox(p + vec3(0.0, 0.0, 0.0), vec3(18.0, 18.5, 18.0));
	//water = matmin(water, vec2(room,2.0));

	//water = matmin(water, vec2(length(p+vec3(0.0,2.5,0.0))-0.5, 1.0));
	//water = matmin(water, vec2(length(p+vec3(2.0,3.6,6.0))-1.0, 0.0));

	float test = 100000.0;// sdBox(p + vec3(2.0,3.6,6.0), vec3(1.0, 0.5, 10.0))+noise*2.0;
	//test = sdTorusFlat(p + vec3(0.0,3.6,0.0)+noise*1.0, vec2(4.5, 5.0), 0.25);
	test = smin(test, sdTorusFlat(p.xzy + vec3(0.0, 3.6, 0.0).xzy + noise, vec2(4.0, 4.5), 0.25), -1.5);
	float sphere = length(p + vec3(0, -0.65, 0)) - 0.5;
	test = smin(test, sphere, -5.0);
	//    water = matmin(water, vec2(sphere, 0.0));
	test = max(test, -sdTorusFlat(p.xzy + vec3(0.0, 3.6, 0.0).xzy + noise, vec2(4.2, 4.3), 0.125));
	water = vec2(test, 0.0);
	float cut = -sdBox(p + vec3(0.0, 0.0, 0.0), vec3(1.0, 1.5, 2.0));
	cut = -(length(p.xy - vec2(0, 0.65)) - 0.5);
	water = matmax(water, vec2(cut, 0.0));
	test = sdTorusFlat(p.xzy + vec3(0.0, 3.6, 0.0).xzy + noise, vec2(4.24, 4.26), 0.03);
	water = matmin(water, vec2(test, 5.0));
	// gold ring
	test = sdTorusFlat(p.xzy + vec3(0.0, -0.65, 0.45).xzy, vec2(.45, .51), 0.05);
	water = matmin(water, vec2(test, 2.0));
	test = sdTorusFlat(p.xzy + vec3(0.0, -0.65, 0.3).xzy, vec2(.0, .42), 0.2);
	water = matmin(water, vec2(test, 1.0));

	test = length(p + vec3(0.0, 3.6, 0.0)) - 3.;
	float n2 = noise - 0.05;
	test = -smin(-test, n2, -5.0);
	float ground = p.y + 4.01;
	ground = sdTorusFlat(p + vec3(0.0, 4.1, 0.0), vec2(0.0, 3.5), 0.005);
	test = smin(test, ground, -9.0);
	//test = p.y+4.01;
	water = matmin(water, vec2(test, 3.0));

	float ring = sdTorusFlat(p + vec3(0.0, 4.1, 0.0), vec2(3.5, 3.5125), 0.00125);
	//water = matmin(water, vec2(ring, 5.0));

	vec3 p2 = RotateX(p, -0.25) + vec3(4.5, 0.0, 4.0);
	float d = sdBox(p2, vec3(4.0, 1.5, 4.0));
	d = max(d, (p2.y + (p2.x * p2.x + p2.z*p2.z) * 0.3)*0.25);
	//water = matmin(water, vec2(d, 1.0));

	//float prism = sdTriPrism(p + vec3(0.0, 3.0, 0.0), vec2(1.0, 6.0));
	//water = matmin(water, vec2(prism, 0.0));

	return water * vec2(float(outside), 1.0);
}

SceneHit SceneMarch(const in Ray ray) {
	SceneHit result;
	result.hitNormal = vec3(0.0);
	//result.pt = farPlane;
	result.objIndex = BIG_INT;
	result.materialIndex = defaultMaterialIndex;
	vec2 distAndMat = vec2(0.0, -1.0);  // Distance and material
	// ----------------------------- Ray march the scene ------------------------------
	const float maxDepth = 18.0; // farthest distance rays will travel
	const float smallVal = 0.0625*0.125*0.0625;
	result.pt = 0.0;
	const float safety = 1.0;// 0.975;
	// First, escape if we are touching a surface already.
	// Get the ray out of the negative part of the distance field. (rough hack)
	float jump = smallVal;
	for (int i = ZERO_TRICK; i < 16; i++) {  // Weird for loop trick so compiler doesn't unroll loop
		// Step along the ray.
		result.hitPos = (ray.p0 + ray.dirNormalized * result.pt);
		distAndMat = DistanceToObject(result.hitPos, ray.outside);

		if (abs(distAndMat.x) >= smallVal) break;
		// move down the ray a safe amount
		result.pt += jump;//safety;//* float(ray.outside);
		//result.pt += distAndMat.x*2.0;//safety;//* float(ray.outside);
		jump *= 2.0;  // This is not super good. Fix me eventually.
		if (result.pt > maxDepth) break;
	}
	// ray marching time
	for (int i = 500; i >= 0; i--)	// This is the count of the max times the ray actually marches.
	{
		// Step along the ray.
		result.hitPos = (ray.p0 + ray.dirNormalized * result.pt);
		// This is _the_ function that defines the "distance field".
		// It's really what makes the scene geometry. The idea is that the
		// distance field returns the distance to the closest object, and then
		// we know we are safe to "march" along the ray by that much distance
		// without hitting anything. We repeat this until we get really close
		// and then break because we have effectively hit the object.
		distAndMat = DistanceToObject(result.hitPos, ray.outside);

		// If we are very close to the object, let's call it a hit and exit this loop.
		if (abs(distAndMat.x) < smallVal) break;
		// move down the ray a safe amount
		result.pt += distAndMat.x*safety;
		if (i == 0) result.pt = maxDepth + 0.01;
		if (result.pt > maxDepth) break;
	}

	// --------------------------------------------------------------------------------
	// If a ray hit an object, calculate the normal and save the hit info.
	if ((result.pt <= maxDepth) && (result.pt > 0.0))
	{
		float dist = distAndMat.x;
		// calculate the normal from the distance field. The distance field is a volume, so if you
		// sample the current point and neighboring points, you can use the difference to get
		// the normal.
		vec3 smallVec = vec3(smallVal, 0, 0);
		// Normals still point out even if we're on the inside.
		//float mid = DistanceToObject(result.hitPos, 1).x;
		//vec3 normalU = vec3(mid - DistanceToObject(result.hitPos - smallVec.xyy, 1).x,
		//                   mid - DistanceToObject(result.hitPos - smallVec.yxy, 1).x,
		//                   mid - DistanceToObject(result.hitPos - smallVec.yyx, 1).x);
		vec3 normalU = vec3(0.0);
		for (int i = min(0, iFrame); i < 4; i++)
		{
			vec3 e = 0.5773*(2.0*vec3((((i + 3) >> 1) & 1), ((i >> 1) & 1), (i & 1)) - 1.0);
			normalU += e * DistanceToObject(result.hitPos + smallVal * e, 1).x;
		}
		result.hitNormal = normalize(normalU);
		result.objIndex = int(distAndMat.y);
		result.materialIndex = int(distAndMat.y);
	}
	else
	{
		// Our ray trace hit nothing. Set object index to big int.
		result.hitNormal = vec3(0.0);
		result.pt = farPlane;
		result.objIndex = BIG_INT;
		result.materialIndex = BIG_INT;
	}
	return result;
}

vec2 Rotate(vec2 v, float rad)
{
	float cos = cos(rad);
	float sin = sin(rad);
	return vec2(cos * v.x + sin * v.y, -sin * v.x + cos * v.y);
}

float Fractal2d(vec2 p, uint seed0)
{
	//return length(p);
	uint seed = SmallHashA(uint(floor((p.x + 256.0)*0.5)));
	seed ^= SmallHashB(uint(floor((p.y + 256.0)*0.5 + 139.0)*15467.0));
	seed ^= seed0;
	//seed = SmallHashIA(seed);
	////return seed*0.25;
	vec2 pr = p;
	if (length(fract(p*0.5) - 0.5) > 0.4675) return 0.5;
	//float scale = 1.0;
	float iter = 1.0;
	for (int i = 0; i < 9; i++)
	{
		// repeat -1 to 1 region, still scaled to -1 to 1
		pr = fract(pr*0.5 + 0.5)*2.0 - 1.0;

		// length squared
		float len = dot(pr, pr);
		// sorta normalize position - divide by length SQUARED
		float inv = 1.102 / len;
		pr *= inv;
		//pr /= dot(pr, pr);
		pr += vec2(pr.y, -pr.x)*sin(59.72);//iTime);
		//pr += vec2(pr.y, -pr.x)*sin(iTime+float(seed&0xffffu))*0.25;
		//pr = Rotate(pr, iTime*0.3);
		float spin = 1.0;//float(i&1) * 2.0 - 1.0;
		//pr = Rotate(pr, spin*iTime*0.0+float(seed&0xffffu));

		// keep track of normalization
		//scale *= inv;
//        iter += 1.0;
	}
	return length(pr*pr)*0.5;//0.475;
	//float b = abs(length(pr*pr));//*abs(pr.y);///scale;
	//return b*0.51;//pow(b, 0.125);
}


// ---- Main ray trace function ----
// Trace a ray, hit, apply material, save hit, and return the bounced ray
Ray TraceOneRay(const in Ray ray) {
	vec3 diffuse = vec3(1.0);
	vec3 emission = vec3(0.0);

	// Send a ray out into the scene. Combine both ray marching and ray tracing.
	Ray newRay;
	newRay.outside = ray.outside;
	SceneHit resultA = SceneIntersect(ray);
	SceneHit resultB = SceneMarch(ray);
	SceneHit result;
	// Take closest hit.
	if (resultA.pt < resultB.pt) {
		result = resultA;
	}
	else {
		result = resultB;
	}
	result = resultB;// ***** HACK GOT RID OF RAY MARCHING... *****

	float md = max(max(abs(result.hitNormal.x), abs(result.hitNormal.y)), abs(result.hitNormal.z));
	if ((result.materialIndex == 4) && (md > 0.85)) result.materialIndex = 3;
	/*if ((result.hitPos.y < 0.003)|| (result.hitPos.y > 1.997)) {
		result.materialIndex = 2;
		//emission += (1.0-step(Fractal2d(result.hitPos.xz, 5u),0.5))*3.0;
		float pattern = (step(Fractal2d(result.hitPos.xz, 5u),0.5));
		if (pattern < 0.5) result.materialIndex = 3;
	}*/


	vec4 currentScatter = PeekScatter();
	Material material = materials[result.materialIndex % materials.length()];
	// Calculate how far the ray goes before hitting a random scattering particle.
	float lifetime = RayLifetime(currentScatter.w);
	// If we hit an object instead of a scatter-particle or the far-plane...
	if (result.pt < min(farPlane, lifetime)) {

		// Debug normal visualization
		//emission = result.hitNormal * 0.5 + 0.5;
		//newRay.outside = 0;
		//SaveHit(diffuse, emission);
		//return newRay;

		//mat3 basis = MakeBasis(result.hitNormal);
		newRay.p0 = result.hitPos;

		vec4 refMaterial = material.reflectRefract;
		float reflectance = material.refMult;
		float fresnel = refMaterial.z;
		// If reflectance is different for different wavelengths, then let's change from
		// rgb to r, g, or b stochastically while not losing energy. So we will end up
		// tracing a ray representing a single wavelength of light.
		// This can be done unconditionally because if xyz are the same, it doesn't matter anyway.
		float choice = RandFloat();
		if ((refMaterial.x != refMaterial.y) || (refMaterial.y != refMaterial.z)) {
			// If we have already split to single wavelength, don't do it again.
			/*if ( ((refMaterial.x == 0.0) && (refMaterial.y == 0.0)) ||
				((refMaterial.y == 0.0) && (refMaterial.z == 0.0)) ||
				((refMaterial.z == 0.0) && (refMaterial.x == 0.0)) ) {
				// Take the non-zero component as the fresnel value.
				fresnel = dot(refMaterial.xyz, vec3(1.0));
			} else {
				// .333 chance of switching to each single channel - r, g, or b.
				if (choice < 0.33333) {
					fresnel = refMaterial.x;
					diffuse *= vec3(1.0, 0.0, 0.0);
				} else if (choice < 0.66666) {
					fresnel = refMaterial.y;
					diffuse *= vec3(0.0, 1.0, 0.0);
				} else diffuse *= vec3(0.0, 0.0, 1.0);
				diffuse *= 3.0;  // To make up for stochastically dropping 2 out of 3 channels
			}*/
			fresnel = max(max(refMaterial.x, refMaterial.y), refMaterial.z);
			diffuse *= refMaterial.xyz / fresnel;
		}
		// Figure out if we should reflect, or if the ray should go into the object (diffuse or refraction)
		// Schlick's approximation
		float oneMinusCos = 1.0 - saturate(dot(ray.dirNormalized, -result.hitNormal* float(ray.outside)));
		float reflectProb = fresnel + (1.0 - fresnel) * pow(oneMinusCos, 5.0);
		reflectProb *= reflectance;
		if (RandFloat() < reflectProb) {
			//if ((refMaterial.x != refMaterial.y) || (refMaterial.y != refMaterial.z)) diffuse *= refMaterial.xyz * fresnel;
			// reflect
			vec3 reflection = reflect(ray.dirNormalized, result.hitNormal);// * float(ray.outside));
			newRay.dirNormalized = normalize(reflection);
			// Already did the probability of reflection before, so no need to multiply anything.
			//diffuse *= vec3(1.0);
		}
		else {
			//if ((refMaterial.x != refMaterial.y) || (refMaterial.y != refMaterial.z)) diffuse *= 1000000.0;
			if (material.doRefraction) {
				// refract
				float refractionIndex = 1.0 / refMaterial.w;  // 1.33 is water, 1.5 is glass.
				if (ray.outside == -1) refractionIndex = 1.0 / refractionIndex;

				vec3 refraction = refract(ray.dirNormalized, result.hitNormal * float(ray.outside), refractionIndex);
				if (dot(refraction, refraction) > 0.0) {
					// Standard refraction
					newRay.dirNormalized = normalize(refraction);
				}
				else {
					// Special case - total internal reflection.
					// This is where at glancing angles, the surface will act like a mirror.
					// It's what makes fiber optics work. :D
					vec3 reflection = reflect(ray.dirNormalized, result.hitNormal * float(ray.outside));
					newRay.dirNormalized = normalize(reflection);
				}
				if (ray.outside == 1) {
					PushScatter(material.scatter);
				}
				else {
					PopScatter(material.scatter);
				}
				newRay.outside = -ray.outside;
			}
			else {
				// Diffuse light
				// Get a random vector in the hemisphere pointing along the normal.
				vec3 rand = RandPointOnSphere();
				vec3 bounce = rand * sign(dot(result.hitNormal, rand));
				newRay.dirNormalized = bounce;
				// Lambert shading model
				//float intensity = dot(bounce, result.hitNormal);
				diffuse *= material.diffuse;// * intensity;
				emission = material.emission;
				float md = max(max(abs(result.hitNormal.x), abs(result.hitNormal.y)), abs(result.hitNormal.z));
				if ((md < 0.85) && (md > 0.8475)) emission += vec3(0.3, 0.1, 0.3)*60.0;
				vec3 hn = abs(result.hitNormal);
				if ((hn.x > hn.y) && (hn.x > hn.z)) emission *= vec3(1.0, 0.7, 0.4);
				else if ((hn.y > hn.x) && (hn.y > hn.z)) emission *= vec3(0.5, 1.0, 0.1);
				else emission *= vec3(0.9, 0.5, 0.1);
				if ((abs(hn.y) < 0.0125) && (md < 0.85)) emission += vec3(1.9);
				float noise = abs(NoiseGen(result.hitPos.xyz*4.0)*1.0);
				if (result.materialIndex == 5) {
					emission *= pow(noise, 6.0)*2646.0;
				}

				// repeat -1 to 1 region, still scaled to -1 to 1
				vec3 wrap = fract(0.5*result.hitPos + 0.5)*2.0 - 1.0;
				float df = abs(length(wrap) - 0.999);
				//if (df < 0.01) emission += vec3(8.0,10,16);
				df = abs(length(wrap) - 0.25);
				//if (df < 0.01) emission += vec3(8.0,10,16);
				df = abs(length(wrap) - 0.0);
				if (df < 0.15) emission += vec3(8.0, 10, 16)*32.0;
				wrap = vec3(0.0);
				wrap.xz = fract(0.5*(result.hitPos.xz + 1.0) + 0.5)*2.0 - 1.0;
				df = abs(length(wrap) - 0.925);
				if (result.hitNormal.y > 0.5)
					if (df < 0.02) emission += vec3(8.0, 10, 16)*1.93;


				// hack to colorize things randomly
				if (result.materialIndex == 5) {
					//diffuse = Hashf3(uint(result.objIndex*17));
					emission *= max(vec3(0.1), HashVec3(uint(result.objIndex + 37)));
					//float grid = pow(abs(fract(result.hitPos.x*8.0)-0.5)*2.0, 2.3);
					//grid = min(grid, max(0.0, sin(result.hitPos.y*96.0)));
					//emission *= grid*3.0;
				}

				if ((result.hitPos.y < 0.003) || (result.hitPos.y > 1.997)) {
					float pattern = step(Fractal2d(result.hitPos.xz, 5u), 0.5);
					//emission += (1.0-step(Fractal2d(result.hitPos.xz, 5u),0.5))*1.0;
					diffuse = mix(vec3(1.0), vec3(0.1), pattern);
				}
				if (result.materialIndex == 3) {
					// checkerboard because it's a ray tracer. :)
					//diffuse *= float((int(newRay.p0.x+8.0) & 1) ^ (int(newRay.p0.y+8.0) & 1) ^ (int(newRay.p0.z+8.0) & 1) ) * 0.2 + 0.8;
				}
			}

		}
	}
	else {
		if (lifetime < farPlane) {
			// Scattering (fog)
			newRay.p0 = ray.p0 + ray.dirNormalized * lifetime;
			newRay.dirNormalized = RandPointOnSphere();
			diffuse *= currentScatter.xyz;
			//emission = material.emission;
		}
		else {
			// Hit the background image. Let's be done ray tracing.
			emission = GetEnvMap(ray.dirNormalized);
			newRay.outside = 0;  // This terminates the ray.
		}
	}
	// Filtering
	// Filter proportional to how long the ray moves inside the object
	// This can also be done with scattering, but this should converge quicker.
	if (ray.outside == -1) {
		vec3 internal = material.filterDiffuseDensity.xyz;
		diffuse *= pow(internal, vec3(abs(result.pt)));
		emission = material.emission * pow(internal, vec3(abs(result.pt)));
	}

	// Save the ray hit in a list so we can calculate color of the pixel later on.
	SaveHit(diffuse, emission);
	return newRay;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
	SetRandomSeed(fragCoord.xy, iResolution.xy, iFrame);
	vec2 position = (gl_FragCoord.xy / iResolution.xy);
	// read original buffer so we can accumulate pixel values into back it.
	vec4 backpixel = texture(iChannel0, position);
	// If we use the mouse to change the view, reset the pixel accumulation buffer
	if (iMouse.z > 0.0) backpixel = vec4(0.0);

	// ---------------- First, set up the camera rays for ray marching ----------------
	vec2 uv_orig = fragCoord.xy / iResolution.xy * 2.0 - 1.0;
	float zoom = 1.4;
	vec2 uv = uv_orig / zoom;

	// Camera up vector.
	vec3 camUp = vec3(0, 1, 0);

	// Camera lookat.
	vec3 camLookat = vec3(0, 3.91, 0);
	//vec3 camLookat=vec3(0,0.91,0);

	// camera orbit with mouse movement
	float mx = iMouse.x / iResolution.x*PI*2.0 - 0.7;
	float my = -iMouse.y / iResolution.y*10.0;
	vec3 camPos = vec3(cos(my)*cos(mx), sin(my), cos(my)*sin(mx))*(8.2);
	// If mouse is in bottom left corner, then use pre-set camera angle.
	//if ((dot(iMouse.xy, vec2(1.0)) <= 64.0)) camPos = vec3(-4.0, 0.1, 7.1);
	if ((dot(iMouse.xy, vec2(1.0)) <= 64.0)) camPos = vec3(-7.0, 0.1, 7.0)*1.02;

	// Camera setup.
	vec3 camVec = normalize(camLookat - camPos);
	vec3 sideNorm = normalize(cross(camUp, camVec));
	vec3 upNorm = cross(camVec, sideNorm);

	// Depth of field hack... I think the math is all wrong.
	const float depthOfFieldAmount = 0.0003;
	vec2 rg = RandGaussianCircle()*depthOfFieldAmount;
	camPos += sideNorm * rg.x;
	camPos += upNorm * rg.y;
	camVec = normalize(camLookat - camPos);
	sideNorm = normalize(cross(upNorm, camVec));
	upNorm = cross(camVec, sideNorm);

	// More camera setup
	vec3 worldFacing = (camPos + camVec);
	vec3 worldPix = worldFacing + uv.x * sideNorm * (iResolution.x / iResolution.y) + uv.y * upNorm;
	vec3 rayVec = normalize(worldPix - camPos);

	// --------------------------------------------------------------------------------
	vec3 colorSum = vec3(0.0);
	// Loop through for a few samples and average the pixel colors from ray tracing.
	for (int s = ZERO_TRICK; s < NUM_SAMPLES; s++) {  // Weird for loop trick so compiler doesn't unroll loop
		InitScatterStack(globalScatter);
		ResetColorHitList();
		Ray ray;
		ray.outside = 1;
		ray.p0 = camPos;
		// Anti-aliasing: Randomly jitter the ray direction by a gaussian distribution.
		vec2 gauss = RandGaussianCircle();
		float antialias = dFdx(uv.xy).x / 1.5;
		ray.dirNormalized = normalize(rayVec +
			sideNorm * gauss.x*antialias +
			upNorm * gauss.y*antialias);

		// Trace a ray from the camera outwards, bounce the ray off objects and keep
		// tracing until NUM_ITERS or until it hits the background.
		for (int i = ZERO_TRICK; i < NUM_ITERS; i++) {
			if (i == (NUM_ITERS - 1)) break;
			ray = TraceOneRay(ray);
			if (ray.outside == 0) break;
		}
		/*int i = 0;
		do {
			ray = TraceOneRay(ray);
			i++;
		} while ((ray.outside != 0) && (i < NUM_ITERS-1));*/

		// Once we're done iterating through rays from the camera outwards, we have a
		// list of hits. Now we can go from the light source toward the camera and apply
		// the color filters and emissions as we go.
		vec4 finalColor = vec4(0.0, 0.0, 0.0, 0.0);
		for (int i = NUM_ITERS - 1; i >= ZERO_TRICK; i--) {
			if (colorHits[i].emission.x != -1.0) {
				finalColor.xyz *= colorHits[i].diffuse;
				finalColor.xyz += colorHits[i].emission;
				//finalColor.xyzw = finalColor.yzwx;  // Debug ray depth
			}
		}
		colorSum += finalColor.xyz;
	}
	colorSum /= float(NUM_SAMPLES);

	// output the final color
	fragColor = vec4(saturate(colorSum / 4096.0), 1.0) + backpixel;
}


