import os
import threading
import time
from pynvml import (
    nvmlInit,
    nvmlShutdown,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetComputeRunningProcesses,
    nvmlDeviceGetMemoryInfo,
)

stop_signal = threading.Event()
monitor_thread = None

def report_actual_gpu_memory(device):
    if device.type != "cuda":
        print("[report_actual_gpu_memory] Device is CPU. Skipping GPU memory report.")
        return

    device_index = device.index if device.index is not None else 0

    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(device_index)
        info = nvmlDeviceGetComputeRunningProcesses(handle)
        for p in info:
            used_mb = p.usedGpuMemory / 1024**2
            print(f"PID {p.pid}: {used_mb:.2f} MB")
    except Exception as e:
        print(f"[report_actual_gpu_memory] Error: {e}")
    finally:
        try:
            nvmlShutdown()
        except:
            pass
def gpu_monitor_loop(device, interval: float = 5.0):
    if device.type != "cuda":
        print("[GPU Monitor] Device is CPU. Skipping GPU monitoring.")
        return

    device_index = device.index if device.index is not None else 0

    while not stop_signal.is_set():
        try:
            nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(device_index)
            mem_info = nvmlDeviceGetMemoryInfo(handle)
            used = mem_info.used / 1024**2
            total = mem_info.total / 1024**2
            print(f"[GPU {device_index}] Used: {used:.2f} MB / {total:.2f} MB")
            report_actual_gpu_memory(device)
        except Exception as e:
            print(f"[GPU Monitor] Error: {e}")
        finally:
            try:
                nvmlShutdown()
            except:
                pass
        time.sleep(interval)

def start_gpu_monitor(device, interval: float = 5.0):
    global monitor_thread

    if os.getenv("MONITOR_GPU", "false").lower() != "true":
        print("⚠️ GPU monitoring is disabled (MONITOR_GPU != true)")
        return

    if device.type != "cuda":
        print("⚠️ GPU monitoring skipped: device is not 'cuda'")
        return

    if monitor_thread and monitor_thread.is_alive():
        print("ℹ️ GPU monitor is already running.")
        return

    print("✅ Starting GPU memory monitor for CUDA device 0...")
    stop_signal.clear()
    monitor_thread = threading.Thread(
        target=gpu_monitor_loop, args=(device, interval, 0), daemon=True
    )
    monitor_thread.start()

def stop_gpu_monitor():
    global monitor_thread
    print("🛑 Stopping GPU memory monitor...")
    stop_signal.set()
    if monitor_thread:
        monitor_thread.join()
        monitor_thread = None
