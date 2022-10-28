from ec2_metadata import ec2_metadata

def get_instance_type():
    try:
        return ec2_metadata.instance_type
    except:
        logger.info("Not an AWS host; returning empty host name")
        return None
