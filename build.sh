cmake -S . -B ./build \
    -DCMAKE_PREFIX_PATH=/home/yiyin.zjh/.local/lib/python3.10/site-packages/torch/share/cmake/Torch/ \
    -DUSE_PER_THREAD_STREAM=ON \
    && cmake --build ./build -j64