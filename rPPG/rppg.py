# IMPORT LIBRARY
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from scipy.signal import butter, filtfilt, detrend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import warnings

warnings.filterwarnings('ignore')


# PARAMETER KONFIGURASI
BUFFER_SECONDS = 30       
WINDOW_SECONDS = 10       
UPDATE_INTERVAL = 0.5     

ASSUME_FPS = 30.0

# ROI Parameters
ROI_SELECTION = "combined"  # Options: "forehead", "cheeks", "nose", "combined"
WIDTH_RATIO = 0.18
HEIGHT_PX = 60
PAD_Y = -8

# Signal Processing Parameters
FS_TARGET = 30.0          
MOVING_AVG_FRAMES = 30    
BANDPASS_LOW = 0.67       
BANDPASS_HIGH = 4.0       

# BPM Smoothing
BPM_SMOOTH_WINDOW = 5
BPM_MIN_VALID = 30
BPM_MAX_VALID = 220

# Visualization
SHOW_PLOT = True          
PLOT_UPDATE_INTERVAL = 2.0  


# INISIALISASI MEDIAPIPE
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


# LANDMARK & ROI
def get_first_valid_landmark(landmarks, candidates, img_w, img_h):
    for idx in candidates:
        try:
            lm = landmarks[idx]
            x = int(lm.x * img_w)
            y = int(lm.y * img_h)
            return x, y
        except Exception:
            continue
    return int(img_w * 0.5), int(img_h * 0.18)

# Landmarks forehead 
def landmarks_to_forehead_box(landmarks, img_w, img_h,
                              width_ratio=WIDTH_RATIO,
                              height_px=HEIGHT_PX,
                              pad_y=PAD_Y):
    left_candidates = [70, 63, 105]  
    right_candidates = [300, 293, 334]  
    
    lx, ly = get_first_valid_landmark(landmarks, left_candidates, img_w, img_h)
    rx, ry = get_first_valid_landmark(landmarks, right_candidates, img_w, img_h)
    
    center_x = int((lx + rx) / 2)
    brows_mean_y = int((ly + ry) / 2)
    bottom_y = int(brows_mean_y + pad_y)
    top_y = bottom_y - height_px
    
    dist_alis = abs(rx - lx)
    width_default = int(max(int(img_w * width_ratio), int(dist_alis * 0.9)))
    half_w = width_default // 2
    
    left_x = center_x - half_w
    right_x = center_x + half_w
    
    left_x = int(np.clip(left_x, 0, img_w - 1))
    right_x = int(np.clip(right_x, 0, img_w - 1))
    top_y = int(np.clip(top_y, 0, img_h - 1))
    bottom_y = int(np.clip(bottom_y, 0, img_h - 1))
    
    pts = np.array([[left_x, top_y],
                    [right_x, top_y],
                    [right_x, bottom_y],
                    [left_x, bottom_y]], dtype=np.int32)
    pts = cv2.convexHull(pts)
    return pts

# Landmarks cheeks
def landmarks_to_cheeks_box(landmarks, img_w, img_h, cheek_side='both'):
    left_cheek_candidates = [226, 113, 50, 2]   
    right_cheek_candidates = [446, 343, 280, 398]  
    
    rois = []
    
    if cheek_side in ['left', 'both']:
        lx, ly = get_first_valid_landmark(landmarks, left_cheek_candidates, img_w, img_h)
        
        # ROI di tengah bagian pipi
        box_size = int(img_h * 0.10)  
        left_x = max(0, lx - box_size // 2.8) 
        right_x = min(img_w - 1, lx + box_size // 2.8)
        top_y = max(0, ly + box_size // 4)  
        bottom_y = min(img_h - 1, ly + box_size // 0.9)  
        
        pts = np.array([[left_x, top_y],
                        [right_x, top_y],
                        [right_x, bottom_y],
                        [left_x, bottom_y]], dtype=np.int32)
        rois.append(cv2.convexHull(pts))
    
    if cheek_side in ['right', 'both']:
        rx, ry = get_first_valid_landmark(landmarks, right_cheek_candidates, img_w, img_h)
      
        # ROI di tengah bagian pipi
        box_size = int(img_h * 0.10)  
        left_x = max(0, rx - box_size // 2.8)  
        right_x = min(img_w - 1, rx + box_size // 2.8)
        top_y = max(0, ry + box_size // 4)  
        bottom_y = min(img_h - 1, ry + box_size // 0.9)  
        
        pts = np.array([[left_x, top_y],
                        [right_x, top_y],
                        [right_x, bottom_y],
                        [left_x, bottom_y]], dtype=np.int32)
        rois.append(cv2.convexHull(pts))
    
    return rois

# Landmarks nose
def landmarks_to_nose_box(landmarks, img_w, img_h):
    nose_candidates = [6, 195, 209, 198, 131]  
    
    nx, ny = get_first_valid_landmark(landmarks, nose_candidates, img_w, img_h)
    
    # Buat box around nose bridge 
    box_width = int(img_h * 0.08)   
    box_height = int(img_h * 0.12)  
    
    left_x = max(0, nx - box_width // 2)
    right_x = min(img_w - 1, nx + box_width // 2)
    top_y = max(0, ny - box_height // 2.5)  
    bottom_y = min(img_h - 1, ny + box_height // 2.5)  
    
    pts = np.array([[left_x, top_y],
                    [right_x, top_y],
                    [right_x, bottom_y],
                    [left_x, bottom_y]], dtype=np.int32)
    pts = cv2.convexHull(pts)
    return pts


def get_roi_boxes(landmarks, img_w, img_h, roi_selection='combined'):
    boxes = []
    names = []
    
    if roi_selection == 'forehead':
        boxes.append(landmarks_to_forehead_box(landmarks, img_w, img_h))
        names.append('Forehead')
    
    elif roi_selection == 'cheeks':
        cheek_boxes = landmarks_to_cheeks_box(landmarks, img_w, img_h, 'both')
        boxes.extend(cheek_boxes)
        names.extend(['Left Cheek', 'Right Cheek'])
    
    elif roi_selection == 'nose':
        boxes.append(landmarks_to_nose_box(landmarks, img_w, img_h))
        names.append('Nose')
    
    elif roi_selection == 'combined':
        boxes.append(landmarks_to_forehead_box(landmarks, img_w, img_h))
        names.append('Forehead')
        cheek_boxes = landmarks_to_cheeks_box(landmarks, img_w, img_h, 'both')
        boxes.extend(cheek_boxes)
        names.extend(['Left Cheek', 'Right Cheek'])
        boxes.append(landmarks_to_nose_box(landmarks, img_w, img_h))
        names.append('Nose')
    
    return boxes, names


# SIGNAL EXTRACTION
def mean_color_in_mask(frame_bgr, mask):
    mask_bool = mask.astype(bool)
    if mask_bool.sum() == 0:
        return None
    
    b, g, r = cv2.split(frame_bgr)
    return float(r[mask_bool].mean()), float(g[mask_bool].mean()), float(b[mask_bool].mean())

# extract signal green channel
def extract_signal_green_channel(times, signal_r, signal_g, signal_b):
    return np.array(signal_g)

# extract signal POS method
def extract_signal_pos(times, signal_r, signal_g, signal_b):
    sig_r = np.array(signal_r)
    sig_g = np.array(signal_g)
    sig_b = np.array(signal_b)
    
    # Normalisasi
    sig_r = sig_r / (np.mean(sig_r) + 1e-8)
    sig_g = sig_g / (np.mean(sig_g) + 1e-8)
    sig_b = sig_b / (np.mean(sig_b) + 1e-8)
    
    sig_pos = (sig_r - sig_b) + (sig_g - sig_r)
    
    return sig_pos


# SIGNAL PROCESSING
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(sig, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, sig)
    return y

def resample_signal_uniform(times, signal, fs_target, window_seconds):
    if len(times) < 2:
        return None, None
    
    t_end = times[-1]
    t_start = t_end - window_seconds
    times_np = np.array(times)
    sig_np = np.array(signal)
    
    mask = times_np >= t_start
    times_in = times_np[mask]
    sig_in = sig_np[mask]
    
    if len(times_in) < 3:
        return None, None
    
    N = int(window_seconds * fs_target)
    t_uniform = np.linspace(times_in[0], times_in[-1], N)
    sig_uniform = np.interp(t_uniform, times_in, sig_in)
    
    return t_uniform, sig_uniform

def detrend_signal(sig, method='moving_avg', moving_avg_frames=30):
    if method == 'moving_avg':
        k = max(3, moving_avg_frames)
        if len(sig) < k:
            k = max(3, int(len(sig) / 4))
        mov = np.convolve(sig, np.ones(k) / k, mode='same')
        return sig - mov
    else:
        return detrend(sig, type=method)


# FFT & BPM ESTIMATION
def estimate_bpm_from_signal(sig, fs, lowcut=BANDPASS_LOW, highcut=BANDPASS_HIGH):
    N = len(sig)
    fft = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    
    idx_band = np.where((freqs >= lowcut) & (freqs <= highcut))[0]
    
    if idx_band.size == 0:
        return None, fft, freqs
    
    fft_band = fft[idx_band]
    freqs_band = freqs[idx_band]
    
    peak_idx = np.argmax(fft_band)
    peak_freq = freqs_band[peak_idx]
    bpm_est = peak_freq * 60.0
    
    if bpm_est < BPM_MIN_VALID or bpm_est > BPM_MAX_VALID:
        return None, fft, freqs
    
    return bpm_est, fft, freqs


# VISUALIZATION (Real-time dengan Matplotlib)
class RealtimePlotter:
    def __init__(self):
        self.fig, self.axes = plt.subplots(3, 1, figsize=(12, 8))
        self.fig.suptitle('rPPG Real-time Analysis')
        
        # Signal plot
        self.ax_signal = self.axes[0]
        self.ax_signal.set_title('Signal (Green Channel)')
        self.ax_signal.set_xlabel('Time (s)')
        self.ax_signal.set_ylabel('Amplitude')
        self.line_signal, = self.ax_signal.plot([], [], lw=2, color='green')
        
        # FFT plot
        self.ax_fft = self.axes[1]
        self.ax_fft.set_title('FFT Spectrum (Bandpass Region)')
        self.ax_fft.set_xlabel('Frequency (Hz)')
        self.ax_fft.set_ylabel('Magnitude')
        self.line_fft, = self.ax_fft.plot([], [], lw=2, color='blue')
        self.ax_fft.set_xlim([BANDPASS_LOW, BANDPASS_HIGH])
        
        # BPM history plot
        self.ax_bpm = self.axes[2]
        self.ax_bpm.set_title('BPM Estimate History')
        self.ax_bpm.set_xlabel('Time')
        self.ax_bpm.set_ylabel('BPM')
        self.line_bpm, = self.ax_bpm.plot([], [], lw=2, marker='o', color='red')
        self.ax_bpm.set_ylim([BPM_MIN_VALID, BPM_MAX_VALID])
        
        plt.tight_layout()
        plt.show(block=False)
        
        self.bpm_history = deque(maxlen=30)
    
    def update(self, t_uniform, sig_detrended, freqs, fft, bpm_est):
        try:
            # Update signal plot
            self.line_signal.set_data(t_uniform, sig_detrended)
            self.ax_signal.set_xlim([t_uniform[0], t_uniform[-1]])
            self.ax_signal.set_ylim([np.min(sig_detrended) * 1.1, np.max(sig_detrended) * 1.1])
            
            # Update FFT plot
            idx_band = np.where((freqs >= BANDPASS_LOW) & (freqs <= BANDPASS_HIGH))[0]
            if idx_band.size > 0:
                self.line_fft.set_data(freqs[idx_band], fft[idx_band])
                self.ax_fft.set_ylim([0, np.max(fft[idx_band]) * 1.2])
            
            # Update BPM history plot
            if bpm_est is not None:
                self.bpm_history.append(bpm_est)
            
            if len(self.bpm_history) > 0:
                self.line_bpm.set_data(range(len(self.bpm_history)), list(self.bpm_history))
                self.ax_bpm.set_xlim([0, 30])
            
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception as e:
            print(f"[WARN] Plot update error: {e}")


# MAIN REAL-TIME LOOP
def run(roi_selection=ROI_SELECTION, signal_method='green', use_plot=SHOW_PLOT):
    print("=" * 70)
    print("Real-time Remote Photoplethysmography (rPPG)")
    print("=" * 70)
    print(f"[CONFIG] ROI Selection: {roi_selection}")
    print(f"[CONFIG] Signal Method: {signal_method}")
    print(f"[CONFIG] Bandpass: {BANDPASS_LOW}-{BANDPASS_HIGH} Hz")
    print(f"[CONFIG] Window: {WINDOW_SECONDS}s, Update: {UPDATE_INTERVAL}s")
    print("=" * 70)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Gagal membuka webcam")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1:
        fps = ASSUME_FPS
    print(f"[INFO] FPS: {fps:.1f} | FS_TARGET: {FS_TARGET} Hz")
    
    # Buffer
    buffer_len = int(BUFFER_SECONDS * max(fps, FS_TARGET))
    times = deque(maxlen=buffer_len)
    signal_r = deque(maxlen=buffer_len)
    signal_g = deque(maxlen=buffer_len)
    signal_b = deque(maxlen=buffer_len)
    
    bpm_history = deque(maxlen=BPM_SMOOTH_WINDOW)
    last_update = time.time()
    last_plot_update = time.time()
    
    # Current BPM display
    current_bpm = 0.0
    
    # Plotter
    plotter = None
    if use_plot:
        try:
            plotter = RealtimePlotter()
        except Exception as e:
            print(f"[WARN] Tidak bisa inisialisasi plotter: {e}")
            use_plot = False
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            h, w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)
            
            # Process face
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0].landmark
                
                # Dapatkan ROI box(es)
                boxes, box_names = get_roi_boxes(face_landmarks, w, h, roi_selection)
                
                # Ekstraksi sinyal dari semua ROI
                total_r, total_g, total_b = 0, 0, 0
                valid_rois = 0
                
                for box in boxes:
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillConvexPoly(mask, box, 255)
                    
                    mean_rgb = mean_color_in_mask(frame, mask)
                    if mean_rgb is not None:
                        r_mean, g_mean, b_mean = mean_rgb
                        total_r += r_mean
                        total_g += g_mean
                        total_b += b_mean
                        valid_rois += 1
                        
                        # Draw ROI
                        cv2.polylines(frame, [box], isClosed=True, color=(0, 255, 0), thickness=2)
                
                # Rata-rata multiple ROI
                if valid_rois > 0:
                    r_mean = total_r / valid_rois
                    g_mean = total_g / valid_rois
                    b_mean = total_b / valid_rois
                    
                    t = time.time()
                    times.append(t)
                    signal_r.append(r_mean)
                    signal_g.append(g_mean)
                    signal_b.append(b_mean)
                    
                    # Display info
                    cv2.putText(frame, f"G: {g_mean:.1f}", (10, 32),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"ROI: {roi_selection} ({valid_rois})",
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # DISPLAY BPM REAL-TIME - Pindah ke bagian bawah layar
                    cv2.putText(frame, f"BPM: {current_bpm:.1f}", (10, h - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            else:
                cv2.putText(frame, "No face detected", (10, 32),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            # Update bpm setiap interval
            now = time.time()
            if (now - last_update) >= UPDATE_INTERVAL and len(times) > 3:
                last_update = now
                
                # Resample ke grid uniform
                t_uniform, sig_resampled = resample_signal_uniform(
                    list(times), list(signal_g), FS_TARGET, WINDOW_SECONDS
                )
                
                if t_uniform is None:
                    cv2.putText(frame, "[WAIT] Buffering...", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                else:
                    # Detrending
                    sig_detrended = detrend_signal(sig_resampled, method='moving_avg',
                                                   moving_avg_frames=MOVING_AVG_FRAMES)
                    
                    # Bandpass filter
                    try:
                        sig_bp = bandpass_filter(sig_detrended, BANDPASS_LOW, BANDPASS_HIGH, FS_TARGET, order=4)
                    except Exception as e:
                        print(f"[WARN] Bandpass filter error: {e}")
                        sig_bp = sig_detrended
                    
                    # FFT & BPM estimation
                    bpm_est, fft, freqs = estimate_bpm_from_signal(sig_bp, FS_TARGET)
                    
                    # Smoothing
                    if bpm_est is not None:
                        bpm_history.append(bpm_est)
                    
                    bpm_smooth = np.mean(bpm_history) if len(bpm_history) > 0 else 0.0
                    current_bpm = bpm_smooth
                    
                    print(f"[BPM] est={bpm_est:.1f} | smooth={bpm_smooth:.1f}")
                    
                    # Update plot
                    if use_plot and plotter and (now - last_plot_update) >= PLOT_UPDATE_INTERVAL:
                        last_plot_update = now
                        plotter.update(t_uniform, sig_detrended, freqs, fft, bpm_est)
            

            
            # Display FPS
            cv2.putText(frame, f"FPS: {fps:.1f}", (w - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow("rPPG Improved - Press 'q' to quit", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()
        if plotter is not None:
            plt.close(plotter.fig)
        print("\n[INFO] Program selesai.")


# ENTRY POINT
if __name__ == "__main__":
    print("\n[OPTIONS]")
    print("ROI Selection: 'forehead', 'cheeks', 'nose', 'combined'")
    print("Signal Method: 'green', 'pos'")
    print("\nDefault: combined ROI, POS method")
    print("\nStarting in 2 seconds...\n")
    
    time.sleep(2)
    
    run(
        roi_selection='combined',
        signal_method='pos',
        use_plot=True
    )