
import numpy as np

def Uint2Sint(num, bits):
    sign_mask = 1 << (bits - 1)
    inv_mask = (1 << (bits)) - 1
    if((num & sign_mask) == 0):
        return num
    else:
        num = num ^ inv_mask
        num = num + 1
        return -num

def NpUint2Sint(x, bits):
    """
    Change a signed/unsigned np.array to unsigned/signed
    :param x: np.array
    :param bits: max bit width of data in x
    :return: signed/unsigned array of x
    """
    sign_mask = 1 << (bits - 1)
    inv_mask = (1 << (bits)) - 1
    x = x.astype(np.int32)
    pos_mask = (x & sign_mask) == 0
    return np.where(pos_mask, x, -((x ^ inv_mask)+1))


if __name__ == "__main__":
    a = 0xfffe
    print(Uint2Sint(a,16))

