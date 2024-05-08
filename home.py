import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import requests

def get_location():
    try:
        latitude = 13.1205561
        longitude = 77.6307445
        response = requests.get(f"https://nominatim.openstreetmap.org/reverse?format=json&lat={latitude}&lon={longitude}")
        data = response.json()
        if 'error' not in data:
            location_name = data.get('display_name', 'Unknown Location')
            return latitude, longitude, location_name
    except Exception as e:
        print("Error fetching location:", e)
    return None, None, None

def split_location_name(location_name):
    words = location_name.split()
    
    num_words = len(words)
    if num_words <= 3:
        return [location_name]
    elif num_words <= 6:
        return [' '.join(words[:3]), ' '.join(words[3:])]
    else:
        line1 = 'Location: '+ ' '.join(words[:3])
        line2 = ' '.join(words[3:6])
        line3 = ' '.join(words[6:])
        return [line1, line2, line3]

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open webcam.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frames = [None] * 100  # Initialize with 100 None values
    motion_events = [0] * 100  # Initialize with 100 zeros
    timestamps = [pd.Timestamp.now() - pd.Timedelta(seconds=i) for i in range(100)]  # Initialize with timestamps for the last 100 seconds

    # Initialize background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from webcam.")
            break

        # Apply background subtraction to detect motion
        fg_mask = bg_subtractor.apply(frame)

        # Count non-zero pixels as motion events
        motion_event_count = np.count_nonzero(fg_mask)

        # Print live stats
        print(f"Live Stats: Motion events - {motion_event_count}")

        cv2.putText(frame, f"Motion events: {motion_event_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Webcam", frame)

        frames.pop(0)  # Remove the oldest frame
        frames.append(frame)  # Add the new frame
        motion_events.pop(0)  # Remove the oldest motion event count
        motion_events.append(motion_event_count)  # Add the new motion event count
        timestamps.pop(0)  # Remove the oldest timestamp
        timestamps.append(pd.Timestamp.now())  # Add the new timestamp

        current_time = time.time()
        if current_time - start_time >= 1:
            start_time = current_time
            timestamps.append(pd.Timestamp.now())
            motion_events.append(motion_event_count)

        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Release the VideoCapture object
    cap.release()
    cv2.destroyAllWindows()

    # Get location information
    latitude, longitude, location_name = get_location()
    if latitude is not None and longitude is not None:
        location_name_lines = split_location_name(location_name)
    else:
        location_name_lines = ["Location information not available"]

    # Plot motion detection data
    plt.figure(figsize=(10, 8))
    plt.plot(timestamps, motion_events, marker='o', color='b')

    # Add red markers at detection times
    for i, event_count in enumerate(motion_events):
        if event_count > 0:
            plt.plot(timestamps[i], event_count, marker='o', markersize=2, color='r')  # Adjust markersize here

    # Add location name to the plot
    for i, line in enumerate(location_name_lines):
        plt.text(0.98, 0.98 - i*0.03, line, transform=plt.gcf().transFigure, fontsize=10, verticalalignment='top', horizontalalignment='right', fontname='Georgia')

    # Adjust x-axis and y-axis fonts
    plt.xticks(fontname='Garamond')
    plt.yticks(fontname='Garamond')

    # Set plot background color to gray
    plt.gca().set_facecolor('lightgray')

    plt.title('Motion Detection Events Over Time', fontname='Georgia')
    plt.xlabel('Time', fontname='Garamond')
    plt.ylabel('Number of Events', fontname='Garamond')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
