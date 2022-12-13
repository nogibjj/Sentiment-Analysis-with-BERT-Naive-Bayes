# import numpy as np
from generative_example import get_real_data_preprocessed

def main():
    """Get real data."""
    X, y = get_real_data_preprocessed()
    print(X.shape)
    print(y.shape)
    print(X[0])
    print(y[0])

if __name__ == "__main__":
    main()