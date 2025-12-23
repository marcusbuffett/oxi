export LIBTORCH_USE_PYTORCH=1
TORCH_LIB=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))/lib")
export DYLD_LIBRARY_PATH="$TORCH_LIB:$DYLD_LIBRARY_PATH"
echo "libtorch → PyTorch (MPS) enabled"
echo "DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH"
