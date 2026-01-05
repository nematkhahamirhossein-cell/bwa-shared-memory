#include "cuda_profiler.cuh"
#include <map>
#include <string>
#include <vector>
#include <mutex>
#include <iostream>

struct KernelStat {
    std::string name;
    float total_ms;
    int launches;
    std::vector<cudaEvent_t> start_events;
    std::mutex mtx;
    KernelStat(): total_ms(0.0f), launches(0) {}
};

static std::map<std::string, KernelStat> g_stats;
static std::mutex g_stats_mtx;
static bool g_inited = false;

void cuda_profiler_init()
{
    if (g_inited) return;
    g_inited = true;
    atexit(cuda_profiler_print_results);
}

void cuda_profiler_record_start(const char* name, cudaStream_t stream)
{
    cuda_profiler_init();
    std::string sname(name);
    std::lock_guard<std::mutex> g(g_stats_mtx);
    auto &stat = g_stats[sname];
    stat.name = sname;
    cudaEvent_t ev;
    cudaEventCreate(&ev);
    cudaEventRecord(ev, stream);
    stat.start_events.push_back(ev);
}

void cuda_profiler_record_end(const char* name, cudaStream_t stream)
{
    std::string sname(name);
    cudaEvent_t ev_end;
    cudaEventCreate(&ev_end);
    cudaEventRecord(ev_end, stream);

    // match with earliest start event
    std::lock_guard<std::mutex> g(g_stats_mtx);
    auto it = g_stats.find(sname);
    if (it==g_stats.end()){
        // no start found, still count it
        KernelStat ks;
        ks.name = sname;
        ks.launches = 1;
        // synchronize end and discard
        cudaEventSynchronize(ev_end);
        cudaEventDestroy(ev_end);
        g_stats[sname] = ks;
        return;
    }
    KernelStat &stat = it->second;
    if (stat.start_events.empty()){
        // nothing to pair, just count
        cudaEventSynchronize(ev_end);
        cudaEventDestroy(ev_end);
        stat.launches += 1;
        return;
    }
    // pop the oldest start
    cudaEvent_t ev_start = stat.start_events.front();
    stat.start_events.erase(stat.start_events.begin());
    // wait for end
    cudaEventSynchronize(ev_end);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev_start, ev_end);
    stat.total_ms += ms;
    stat.launches += 1;
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_end);
}

void cuda_profiler_print_results()
{
    std::lock_guard<std::mutex> g(g_stats_mtx);
    std::cerr << "\n=== CUDA Kernel Profiling Results ===\n";
    std::cerr << "Kernel Name, Total Time (ms), Launches" << std::endl;
    for (auto &p : g_stats){
        const KernelStat &s = p.second;
        std::cerr << s.name << ", " << s.total_ms << ", " << s.launches << std::endl;
    }
    std::cerr << "=== End Profiling Results ===\n";
}
