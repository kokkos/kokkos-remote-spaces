#ifndef RACERLIB_INTERFACE_H
#define RACERLIB_INTERFACE_H

#include <RDMA_engine.hpp>
#include <Kokkos_Core.hpp>

namespace RACERlib{

struct Engine{
    RdmaScatterGatherEngine* sge;
    RdmaScatterGatherWorker* sgw;

    static std::set<RdmaScatterGatherEngine*> sges;

    Engine(){}

    void allocate_device(void * p, MPI_Comm comm)
    {
        size_t header_size = 0x1;
        sge = allocate_rdma_scatter_gather_engine(p, sizeof(uint64_t), header_size, comm);
        sges.insert(sge);
        cudaMalloc(&sgw, sizeof(RdmaScatterGatherWorker));
        RdmaScatterGatherWorker dev_worker;
        dev_worker.tx_element_request_ctrs = sge->tx_element_request_ctrs; //already a device buffer
        dev_worker.rx_element_reply_queue = sge->rx_element_reply_queue_mr->addr; //already a device buffer
        dev_worker.tx_element_reply_queue = sge->tx_element_reply_queue_mr->addr;
        dev_worker.tx_element_request_trip_counts = sge->tx_element_request_trip_counts; //already a device buffer
        dev_worker.cache = sge->cache;
        dev_worker.direct_ptrs = sge->direct_ptrs_d; //already a device buffer
        dev_worker.tx_element_request_queue = (uint32_t*) sge->tx_element_request_queue_mr->addr;
        dev_worker.ack_ctrs_d = sge->ack_ctrs_d;
        dev_worker.tx_element_reply_ctrs = sge->tx_element_reply_ctrs;
        dev_worker.rx_element_request_queue = (uint32_t*)sge->rx_element_request_queue_mr->addr;
        dev_worker.num_pes = sge->num_pes;
        dev_worker.tx_element_aggregate_ctrs = sge->tx_element_aggregate_ctrs;
        dev_worker.tx_block_request_cmd_queue = sge->tx_block_request_cmd_queue;
        dev_worker.tx_block_reply_cmd_queue = sge->tx_block_reply_cmd_queue;
        dev_worker.tx_block_request_ctr = sge->tx_block_request_ctr;
        dev_worker.rx_block_request_cmd_queue = sge->rx_block_request_cmd_queue;
        dev_worker.rx_block_request_ctr = sge->rx_block_request_ctr;
        dev_worker.tx_element_request_ctrs = sge->tx_element_request_ctrs;
        cuda_safe(cuMemHostGetDevicePointer((CUdeviceptr *)&dev_worker.ack_ctrs_h, sge->ack_ctrs_h, 0));
        dev_worker.rank = sge->rank;
        dev_worker.request_done_flag = sge->request_done_flag;
        dev_worker.response_done_flag = sge->response_done_flag;
        cuda_safe(cuMemHostGetDevicePointer((CUdeviceptr *)&dev_worker.fence_done_flag, sge->fence_done_flag, 0));
        cudaMemcpyAsync(sgw, &dev_worker, sizeof(RdmaScatterGatherWorker), cudaMemcpyHostToDevice);
    }

    void allocate_host()
    {
        sgw = new RdmaScatterGatherWorker;
        sgw->tx_element_request_ctrs = sge->tx_element_request_ctrs;
        sgw->ack_ctrs_h = sge->ack_ctrs_h;
        sgw->tx_element_request_queue = (uint32_t*) sge->tx_element_request_queue_mr->addr;
        sgw->rx_element_reply_queue = sge->rx_element_reply_queue_mr->addr;
        sgw->rank = sge->rank;
    }



    RdmaScatterGatherWorker* get_worker() const {
        return sgw;
    }

    RdmaScatterGatherEngine* get_engine() const {
        return sge;
    }


    ~Engine(){
        fence();
        sges.erase(sge);
        free_rdma_scatter_gather_engine(sge);
    }

    void fence()
    {
        for (RdmaScatterGatherEngine* engine : sges){
            engine->fence();
        }
    }
};


template <typename Type, typename Traits>
void put(void * comm_id,void * allocation, Type & value, int PE, int offset);
template <typename Type, typename Traits>
Type get(void * comm_id, void * allocation, int PE, int offset); 

void start(void * comm_id, void * allocation_id);
void stop(void * comm_id, void * allocation_id);
void flush(void * comm_id, void * allocation);
int init(void * comm_id); //set communicator reference, return RACERLIB_STATUS
int finalize (void * comm_id); //finalize communicator instance, return RECERLIB_STATUS


/*
Kokkos::MemoryTraits<Kokkos::Cached>
Kokkos::MemoryTraits<Kokkos::Aggregated>



*/



void start(void * comm_id, void * allocation_id)
{
  //  allocate_device();
   // allocate_host();


}


} // namespace RACERlib

#endif