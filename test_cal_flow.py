┌───────────────────────────────────┐
│ Extract and Match Features        │
│                                   │
│ - Use ORB to extract keypoints.   │
│ - Use BFMatcher/FLANN for matching.     │
│ - Filter good matches.            │
└───────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│ Generate 3D-2D Mappings           │
│                                   │
│ - Map matched 2D points to 3D     │
│   object points.                  │
│ - If points < 6, replicate data.  │
└───────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│ Estimate Camera Pose (PnP)        │
│                                   │
│ - SolvePnP to compute rvec, tvec. │
│ - Convert rvec to rotation matrix.│
│ - Calculate Yaw, Pitch, Roll.     │
└───────────────────────────────────┘