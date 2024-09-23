import cupy as cp
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def check_cuda():
    try:
        logging.info("Checking CUDA availability")
        device_count = cp.cuda.runtime.getDeviceCount()
        logging.info(f"Number of CUDA devices: {device_count}")

        for i in range(device_count):
            device = cp.cuda.Device(i)
            props = cp.cuda.runtime.getDeviceProperties(i)
            logging.info(f"Device {i}: {props['name'].decode()}")
            logging.info(f"  Compute Capability: {props['major']}.{props['minor']}")
            logging.info(f"  Total Memory: {props['totalGlobalMem'] / 1e9:.2f} GB")

        current_device = cp.cuda.runtime.getDevice()
        logging.info(f"Current CUDA device: {current_device}")

        # Try to perform a simple CUDA operation
        a = cp.array([1, 2, 3])
        b = cp.array([4, 5, 6])
        c = a + b
        logging.info(f"CUDA operation result: {c}")

        logging.info("CUDA is available and working correctly")
        return True

    except Exception as e:
        logging.error(f"CUDA initialization error: {str(e)}")
        return False

if __name__ == "__main__":
    if check_cuda():
        logging.info("CUDA check completed successfully")
    else:
        logging.error("CUDA check failed")