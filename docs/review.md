# Synchronized review exports

`Render -ReviewManifest ... -ViewerDescriptor ... -ViewerXml ...
-ReviewSourceComparison` reads saved compact motion with native meshes and exact
source timing. It does not alter or regenerate the motion.

`Gallery -GalleryClips ... -GalleryHideUI -GalleryContactGlow` exports scene-only
comparison GIFs. Add `-GallerySeparate` for one GIF per action; width and duration
are configurable. Green contact regions and ground grid remain visible.
Do not judge published motion validity from solving success alone. Inspect
source/robot timing, left/right wrist direction, whole-body posture and contacts.
No generated review assets are part of this source repository.
