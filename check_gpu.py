import tensorflow as tf
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_tensorflow_gpu():
    logger.info(f"TensorFlow version: {tf.__version__}")

    # Check for GPU availability
    gpus = tf.config.list_physical_devices('GPU')
   
    if gpus:
        logger.info(f"Number of GPUs available: {len(gpus)}")
        for gpu in gpus:
            logger.info(f"GPU name: {gpu.name}, Type: {gpu.device_type}")
       
        # Test GPU with a simple operation
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c = tf.matmul(a, b)
        
        logger.info(f"GPU computation test result: {c.numpy()}")
    else:
        logger.error("No GPU available. Program will not run without GPU.")
        sys.exit(1)

    # Check for CUDA availability
    cuda_available = tf.test.is_built_with_cuda()
    logger.info(f"CUDA available: {cuda_available}")

    # Additional TensorFlow build information
    logger.info(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    logger.info(f"Built with ROCm: {tf.test.is_built_with_rocm()}")

if __name__ == "__main__":
    check_tensorflow_gpu()
