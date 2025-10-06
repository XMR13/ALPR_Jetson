import numpy as np
import pandas as pd

def testing_library():
    """Mengetes library apakah bisa atau tidak"""
    xx = np.array([1,2,3,4,5,6,7])
    print(F"arraynya adalah {xx}")

    #mengetes untuk bagian pandas
    print(f"versi library pandas adalah {pd.__version__}")

def main():
    print("Hello from alpr-jetson!")
    testing_library()


if __name__ == "__main__":
    main()
