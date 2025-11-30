# Label Studio Export Guide

This app parses Label Studio JSON exports containing annotations for a video or frame sequence. Supported prompts:
- RectangleLabels (bounding boxes)
- Points / KeyPointLabels (point clicks)
- BrushLabels / pixel masks (RLE/polygons where available)

Example (simplified):
```json
[
  {
    "id": 123,
    "data": { "video": "s3://bucket/video.mp4", "frameCount": 120 },
    "annotations": [
      {
        "id": 456,
        "result": [
          {
            "id": "rect-1",
            "type": "rectanglelabels",
            "value": {
              "x": 20.5, "y": 30.2, "width": 15.0, "height": 10.0, "frame": 12,
              "rectanglelabels": ["ObjectA"]
            },
            "original_width": 1920, "original_height": 1080
          },
          {
            "id": "pt-1",
            "type": "keypointlabels",
            "value": { "x": 42.2, "y": 55.1, "frame": 12, "keypointlabels": ["ObjectA"] },
            "original_width": 1920, "original_height": 1080
          },
          {
            "id": "mask-1",
            "type": "brushlabels",
            "value": { "rle": "encodedRLEHere", "frame": 12, "brushlabels": ["ObjectA"] },
            "original_width": 1920, "original_height": 1080
          }
        ]
      }
    ]
  }
]
```

Notes:
- Label Studio uses percentage coords for rectangles and points; this app converts them using `original_width`/`original_height`.
- RLE masks are decoded when provided. If polygons are provided, you may need to adapt `labelstudio_parser.py` to rasterize them.
- `frame` is assumed 0-based; the app normalizes to 1-based frame file numbering.

Troubleshooting:
- If masks are misaligned, verify resolution consistency and indices.
- If multiple tasks exist in the export, the app uses the first with annotations.
