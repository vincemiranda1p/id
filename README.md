# ID Capture Checker (Mobile Web)

A lightweight mobile-first web app that checks live camera frames and only allows capture when:

- An ID-like rectangle is detected in the guide box.
- The ID fits the guide box closely.
- The frame is not blurred/out of focus.
- A face is detected in the ID region.

Real-time issue comments shown on screen:

- `NO ID FOUND`
- `ID NOT IN FRAME`
- `BLURRED / UNCLEAR PHOTO OF WHOLE ID`
- `BLURRY FACE`
- `NO FACE`

Action flow:

- `Submit Photo` button enables only when all checks pass (green rectangle).
- `Submit Photo` captures once, displays the photo, and stops the live camera.
- `Redo Scan` restarts the live camera and lets the user retake.

## Adjusting face blur sensitivity

- Default threshold is in `app.js` as `FACE_BLUR_LAPLACIAN_THRESHOLD_DEFAULT` (currently `45`).
- Higher value = stricter blur check (more likely to flag `BLURRY FACE`).
- Lower value = more lenient blur check.
- Optional runtime override without code edits:
  - `https://your-site-url/?faceBlurThreshold=55`

## Models and libraries used (free + local on device)

- OpenCV.js (image analysis in-browser, on device)
- MediaPipe Face Detector (BlazeFace short-range model) via `@mediapipe/tasks-vision`

These run locally in the browser on modern Android devices (including Samsung A15 class hardware).

## Run locally

1. Open a terminal in this folder.
2. Run a static server:

```powershell
python -m http.server 8080
```

3. Open `http://localhost:8080` on your machine or over LAN from your phone.

## Publish (free)

Use any static host with HTTPS (required for camera permission):

- Netlify Drop: drag-and-drop this folder on [https://app.netlify.com/drop](https://app.netlify.com/drop)
- Cloudflare Pages: connect repo or upload buildless static files
- GitHub Pages: push files to a repo and enable Pages

## Publish on GitHub Pages

This repo now includes a GitHub Pages workflow at `.github/workflows/deploy-pages.yml`.

1. Create a new GitHub repository.
2. Upload or push these files to the `main` branch.
3. In GitHub, open `Settings` -> `Pages`.
4. Under `Build and deployment`, set `Source` to `GitHub Actions`.
5. Push to `main` again if needed, or run the `Deploy GitHub Pages` workflow manually.
6. Wait for the workflow to finish, then open:

```text
https://YOUR_GITHUB_USERNAME.github.io/YOUR_REPOSITORY_NAME/
```

If you want the site at `https://YOUR_GITHUB_USERNAME.github.io/`, name the repository:

```text
YOUR_GITHUB_USERNAME.github.io
```

## Notes

- Camera requires HTTPS on phones (except localhost).
- Good lighting and steady hands improves blur/focus detection.
- If the app remains red, check `Issues:` text for all current problems.
