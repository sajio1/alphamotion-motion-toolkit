"""Geometry-based labels for any robot surface that reaches the ground plane."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def ground_surface_patches(vertices_m, faces, contact_band_cm=1., penetration_cm=.5):
    """Clip actual mesh triangles to floor bands (Y up, meters).

    Returns green and red triangle meshes in world coordinates. Intersections
    are retained even when a triangle centroid sits outside the contact band.
    This is a geometric display proxy; it is not a measured pressure patch.
    """
    vertices = np.asarray(vertices_m, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    upper, lower = contact_band_cm/100, -penetration_cm/100
    if contact_band_cm < 0 or penetration_cm < 0: raise ValueError('Negative threshold')
    def clip(poly, level, below):
        if not len(poly): return []
        result=[]
        for a,b in zip(poly,poly[1:]+poly[:1]):
            ia=(a[1]<=level) if below else (a[1]>=level)
            ib=(b[1]<=level) if below else (b[1]>=level)
            if ia:result.append(a)
            if ia!=ib:
                result.append(a+(b-a)*((level-a[1])/(b[1]-a[1])))
        return result
    triangles=vertices[faces]
    candidates=triangles[triangles[:,:,1].min(1)<=upper]
    meshes=[[],[]]
    for tri in candidates:
        green=clip(clip(list(tri),upper,True),lower,False)
        red=clip(list(tri),lower,True) if tri[:,1].min()<lower else []
        for output,poly in zip(meshes,(green,red)):
            for i in range(1,len(poly)-1):output.extend((poly[0],poly[i],poly[i+1]))
    return tuple((np.asarray(v,np.float32).reshape(-1,3),
                  np.arange(len(v),dtype=np.uint32).reshape(-1,3)) for v in meshes)


def classify_surface_contacts(surface_min_height_cm, contact_band_cm=1.0, penetration_cm=0.5):
    """Return near-ground and penetration masks for ``[frame, surface]`` heights."""
    heights = np.asarray(surface_min_height_cm, dtype=float)
    if heights.ndim != 2:
        raise ValueError("surface_min_height_cm must have shape [frames, surfaces]")
    if contact_band_cm < 0 or penetration_cm < 0:
        raise ValueError("thresholds must be non-negative")
    penetration = heights < -penetration_cm
    contact = (heights <= contact_band_cm) & ~penetration
    return contact, penetration


def export_surface_contacts(surface_min_height_cm, surface_names, fps, output,
                            contact_band_cm=1.0, penetration_cm=0.5):
    """Persist reusable per-frame labels for all visual surfaces, not only feet."""
    heights = np.asarray(surface_min_height_cm, dtype=float)
    names = [str(name) for name in surface_names]
    if heights.shape[1] != len(names):
        raise ValueError("surface_names does not match the height matrix")
    contact, penetration = classify_surface_contacts(heights, contact_band_cm, penetration_cm)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "surface_contact_labels.npz", fps=float(fps),
                        surface_names=np.asarray(names), min_height_cm=heights,
                        contact=contact, penetration=penetration)
    with (output / "surface_contact_labels.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "time_s", "surface", "min_height_cm", "contact", "penetration"])
        for frame in range(len(heights)):
            for index, name in enumerate(names):
                writer.writerow([frame, frame / float(fps), name, heights[frame, index],
                                 int(contact[frame, index]), int(penetration[frame, index])])
    schema = {
        "schema": "greenwich.geometry_surface_contacts.v1",
        "coordinate_frame": "robot render world, ground plane z=0",
        "contact_band_cm": float(contact_band_cm),
        "penetration_cm": float(penetration_cm),
        "surfaces": names,
        "interpretation": "Geometry proximity label; not measured force or dynamic contact.",
    }
    (output / "surface_contact_labels.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return {"contact": contact, "penetration": penetration, "min_height_cm": heights}
