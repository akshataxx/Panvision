"""Train the DenseNet-121 baseline."""
try:
    from .train_baseline import main
except ImportError:
    from train_baseline import main

if __name__ == '__main__':
    main('densenet')
