import sys
import time

def print_library_versions():
    packages = [
        ("Python", lambda: sys.version.split()[0]),
        ("OpenCV (cv2)", lambda: __import__("cv2").__version__),
        ("MediaPipe", lambda: __import__("mediapipe").__version__),
        ("NumPy", lambda: __import__("numpy").__version__),
        ("Pandas", lambda: __import__("pandas").__version__),
        ("Scikit-learn", lambda: __import__("sklearn").__version__),
        ("Matplotlib", lambda: __import__("matplotlib").__version__),
        ("Seaborn", lambda: __import__("seaborn").__version__),
        ("Joblib", lambda: __import__("joblib").__version__),
    ]

    print("=" * 45)
    print("        Installed Library Versions           ")
    print("=" * 45)

    all_imported = True
    for name, getter in packages:
        try:
            ver = getter()
            print(f"  {name:<22} : {ver}")
        except ImportError as e:
            all_imported = False
            print(f"  {name:<22} : NOT INSTALLED ({e})")
        except Exception as e:
            print(f"  {name:<22} : Error ({e})")

    print("=" * 45)
    return all_imported


def test_webcam(duration_seconds=5):
    print(f"\nAttempting to open webcam for {duration_seconds} seconds...")

    try:
        import cv2
    except ImportError:
        print("[ERROR] OpenCV is not installed. Run `pip install -r requirements.txt`.")
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[WARNING] Could not access webcam at index 0.")
        print("          On macOS, ensure Camera permissions are granted to your terminal / IDE.")
        return

    start_time = time.time()
    window_name = "Driver Safety Monitor - Webcam Setup Check (5s)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print("Webcam opened successfully. Showing live preview...")

    while True:
        elapsed = time.time() - start_time
        remaining = max(0.0, duration_seconds - elapsed)

        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to capture frame from webcam.")
            break

        # Display remaining time overlay
        text = f"Closing in: {remaining:.1f}s (Press 'q' to exit)"
        cv2.putText(
            frame,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(window_name, frame)

        # Break early if 'q' is pressed or 5 seconds elapsed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Preview interrupted by user.")
            break

        if elapsed >= duration_seconds:
            print(f"Webcam test completed ({duration_seconds} seconds elapsed).")
            break

    cap.release()
    cv2.destroyAllWindows()
    # macOS OpenCV window cleanup
    for _ in range(5):
        cv2.waitKey(1)


def main():
    print_library_versions()
    test_webcam(duration_seconds=5)


if __name__ == "__main__":
    main()
