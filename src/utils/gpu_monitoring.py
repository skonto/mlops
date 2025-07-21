import os
import threading
import time

from loguru import logger
from pynvml import (
    nvmlDeviceGetComputeRunningProcesses,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlInit,
    nvmlShutdown,
)

stop_signal = threading.Event()
monitor_thread = None


def report_actual_gpu_memory(device):
    if device.type != "cuda":
        logger.info(
            "[report_actual_gpu_memory] Device is CPU. Skipping GPU memory report."
        )
        return

    device_index = device.index if device.index is not None else 0

    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(device_index)
        info = nvmlDeviceGetComputeRunningProcesses(handle)
        for p in info:
            used_mb = p.usedGpuMemory / 1024**2
            logger.info(f"PID {p.pid}: {used_mb:.2f} MB")
    except Exception as e:
        logger.info(f"[report_actual_gpu_memory] Error: {e}")
    finally:
        try:
            nvmlShutdown()
        except:
            pass


def gpu_monitor_loop(device, interval: float = 5.0):
    if device.type != "cuda":
        logger.info("[GPU Monitor] Device is CPU. Skipping GPU monitoring.")
        return

    device_index = device.index if device.index is not None else 0

    while not stop_signal.is_set():
        try:
            nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(device_index)
            mem_info = nvmlDeviceGetMemoryInfo(handle)
            used = mem_info.used / 1024**2
            total = mem_info.total / 1024**2
            logger.info(f"[GPU {device_index}] Used: {used:.2f} MB / {total:.2f} MB")
            report_actual_gpu_memory(device)
        except Exception as e:
            logger.info(f"[GPU Monitor] Error: {e}")
        finally:
            try:
                nvmlShutdown()
            except:
                pass
        time.sleep(interval)


def start_gpu_monitor(device, interval: float = 5.0):
    global monitor_thread

    if os.getenv("MONITOR_GPU", "false").lower() != "true":
        logger.info("GPU monitoring is disabled (MONITOR_GPU != true)")
        return

    if device.type != "cuda":
        logger.info("GPU monitoring skipped: device is not 'cuda'")
        return

    if monitor_thread and monitor_thread.is_alive():
        logger.info("GPU monitor is already running.")
        return

    logger.info("Starting GPU memory monitor for CUDA device 0...")
    stop_signal.clear()
    monitor_thread = threading.Thread(
        target=gpu_monitor_loop, args=(device, interval, 0), daemon=True
    )
    monitor_thread.start()


def stop_gpu_monitor():
    global monitor_thread
    logger.info("Stopping GPU memory monitor...")
    stop_signal.set()
    if monitor_thread:
        monitor_thread.join()
        monitor_thread = None
