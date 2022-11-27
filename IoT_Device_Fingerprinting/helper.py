
def check_polarity(ip_src:str)->int:
    """Check the direction of a packet

    Args:
        ip_src (str): Source IP Address

    Returns:
        int: -1 for incoming packets, 1 for outgoing packets
    """
    if ip_src.startswith('192'):
        return 1
    else:
        return -1

