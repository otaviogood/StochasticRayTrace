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

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
	vec2 uv = fragCoord.xy / iResolution.xy;
	vec4 c = texture(iChannel0, uv);
	fragColor = vec4(sqrt((c.xyz*4096.0) / c.w), 1.0);
}