import urllib.request
import json
import time

def main():
    # 1. Load sample
    req = urllib.request.Request('http://127.0.0.1:5001/api/load_sample', data=b'{}', headers={'Content-Type': 'application/json'})
    res = urllib.request.urlopen(req)
    print('Sample loaded:', res.read().decode())

    # 2. Start pipeline with high-detail RF-DETR Small (conf: 0.22)
    req = urllib.request.Request('http://127.0.0.1:5001/api/pipeline/start', data=json.dumps({'conf': 0.22}).encode(), headers={'Content-Type': 'application/json'})
    res = urllib.request.urlopen(req)
    print('Pipeline start:', res.read().decode())

    # 3. Poll until completed
    for i in range(120):
        time.sleep(1.0)
        res = urllib.request.urlopen('http://127.0.0.1:5001/api/pipeline/status')
        state = json.loads(res.read().decode())
        cf = state.get("current_frame", 0)
        tf = state.get("total_frames", 0)
        pct = state.get("percent", 0.0)
        fps = state.get("fps", 0.0)
        lat = state.get("latency_ms", 0.0)
        phones = state.get("active_tracks_count", 0)
        st = state.get("state", "")
        print(f"[{i:02d}s] Frame: {cf}/{tf} ({pct}%) | FPS: {fps:.1f} | Latency: {lat:.1f}ms | Phones: {phones} | State: {st}")
        if st == 'completed':
            print("\n===============================")
            print("PIPELINE TEST SUCCESSFUL!")
            print(f"Status message: {state.get('status_message')}")
            print(f"Total Unique Tracks: {state.get('total_unique_tracks')}")
            print("===============================\n")
            return
        elif st == 'error':
            print("PIPELINE ERROR:", state.get("error_message"))
            return

if __name__ == '__main__':
    main()
