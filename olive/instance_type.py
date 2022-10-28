from ec2_metadata import ec2_metadata
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_instance_type():
    try:
        return ec2_metadata.instance_type
    except:
        logger.info("Not an AWS host; returning empty host name")
        return None
