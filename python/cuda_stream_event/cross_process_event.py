import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def producer(i):
    dist.init_process_group(
        backend='nccl',
        init_method="tcp://127.0.0.1:39031",
        world_size=2,
        rank=0,
    )
    # 准备数据
    data = torch.randn((1024 * 1024, ), device=f"cuda:{i}")
    func, args = torch.multiprocessing.reductions.reduce_tensor(data)
    args = list(args)
    dist.broadcast_object_list([(func, args)], src=0)
    event = torch.cuda.Event(interprocess=True)
    # 广播 event 的 ipc_handle
    dist.broadcast_object_list([event.ipc_handle()], src=0)
    # 用一个 kernel 来处理数据
    for i in range(100):
        data += 1
    # 记录事件
    event.record()
    # 进程同步，确保 event wait 在 event record 之后
    dist.barrier()
    torch.cuda.synchronize()
    dist.destroy_process_group()


def consumer(j):
    dist.init_process_group(
        backend='nccl',
        init_method="tcp://127.0.0.1:39031",
        world_size=2,
        rank=1,
    )
    torch.cuda.set_device(j)
    # 接收数据
    recv = [None]
    dist.broadcast_object_list(recv, src=0)
    func, args = recv[0]
    args[6] = j
    data = func(*args)
    # 接收 event
    recv = [None]
    dist.broadcast_object_list(recv, src=0)
    event_handle = recv[0]
    event = torch.cuda.Event.from_ipc_handle(device=j, handle=event_handle)
    # 进程同步，确保 event wait 在 event record 之后
    dist.barrier()
    # 等待 event
    event.wait()
    # 判断数据正确性
    assert any([val == 100 for val in data]), "Data is not correct"
    dist.destroy_process_group()


if __name__ == "__main__":
    pi = mp.Process(target=producer, args=(0,))
    pj = mp.Process(target=consumer, args=(1,))
    pi.start()
    pj.start()
    pi.join()
    pj.join()
    assert pi.exitcode == 0 and pj.exitcode == 0, "Process failed"

