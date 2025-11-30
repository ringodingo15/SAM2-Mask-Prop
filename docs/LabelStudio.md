# Label Studio Export Guide

This app parses Label Studio JSON exports containing annotations for a video or frame sequence. It supports three prompt types:

- RectangleLabels (bounding boxes)
- Points / KeyPointLabels (point clicks)
- BrushLabels / pixel masks (RLE/polygons where available)

Below is a simplified example of a single-task JSON entry:

```json
[
  {
    "id": 123,
    "data": {
      "video": "s3://bucket/video.mp4",
      "frameCount": 120
    },
    "annotations": [
      {
        "id": 456,
        "result": [
          {
            "id": "rect-1",
            "type": "rectanglelabels",
            "value": {
              "x": 20.5,
              "y": 30.2,
              "width": 15.0,
              "height": 10.0,
              "rotation": 0,
              "rectanglelabels": ["ObjectA"],
              "frame": 12
            },
            "from_name": "bbox",
            "to_name": "video",
            "original_width": 1920,
            "original_height": 1080
          },
          {
            "id": "pt-1",
            "type": "keypointlabels",
            "value": {
              "x": 42.2,
              "y": 55.1,
              "keypointlabels": ["ObjectA"],
              "frame": 12
            },
            "from_name": "points",
            "to_name": "video",
            "original_width": 1920,
            "original_height": 1080
          },
          {
            "id": "mask-1",
            "type": "brushlabels",
            "value": {
              "rle": "encodedRLEHere",
              "brushlabels": ["ObjectA"],
              "frame": 12
            },
            "from_name": "mask",
            "to_name": "video",
            "original_width": 1920,
            "original_height": 1080
          }
        ]
      }
    ]
  }
]
```

Notes:
- Rectangle coordinates `x`, `y`, `width`, `height` are percentages in Label Studio—this app converts them using `original_width` and `original_height`.
- Points use `x`, `y` in percentages as well.
- Masks may be provided as `rle` or sometimes polygons; the app decodes RLE into a binary mask aligned to frame dimensions.
- Frame indices: `frame` should indicate the annotated frame index. Ensure it corresponds to your extracted frames (0- or 1-based indexing can vary; the app attempts to normalize but you may need to adjust in config if your export is unusual).

Workflow:
1) Export JSON from your Label Studio project.
2) In the app UI, upload that JSON using “Upload Label Studio JSON”.
3) The server extracts per-frame prompts grouped by object label.

If multiple tasks are included in one JSON export, the app picks the first relevant one by default. Future UI updates may allow selecting the task explicitly.

Troubleshooting:
- If the app can’t find frames for your annotations, verify your frame numbering and `frame` indices in JSON.
- If masks look misaligned, check `original_width`/`original_height` and confirm your frames share that resolution.
