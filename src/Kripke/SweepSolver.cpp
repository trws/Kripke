/*
 * NOTICE
 *
 * This work was produced at the Lawrence Livermore National Laboratory (LLNL)
 * under contract no. DE-AC-52-07NA27344 (Contract 44) between the U.S.
 * Department of Energy (DOE) and Lawrence Livermore National Security, LLC
 * (LLNS) for the operation of LLNL. The rights of the Federal Government are
 * reserved under Contract 44.
 *
 * DISCLAIMER
 *
 * This work was prepared as an account of work sponsored by an agency of the
 * United States Government. Neither the United States Government nor Lawrence
 * Livermore National Security, LLC nor any of their employees, makes any
 * warranty, express or implied, or assumes any liability or responsibility
 * for the accuracy, completeness, or usefulness of any information, apparatus,
 * product, or process disclosed, or represents that its use would not infringe
 * privately-owned rights. Reference herein to any specific commercial products,
 * process, or service by trade name, trademark, manufacturer or otherwise does
 * not necessarily constitute or imply its endorsement, recommendation, or
 * favoring by the United States Government or Lawrence Livermore National
 * Security, LLC. The views and opinions of authors expressed herein do not
 * necessarily state or reflect those of the United States Government or
 * Lawrence Livermore National Security, LLC, and shall not be used for
 * advertising or product endorsement purposes.
 *
 * NOTIFICATION OF COMMERCIAL USE
 *
 * Commercialization of this product is prohibited without notifying the
 * Department of Energy (DOE) or Lawrence Livermore National Security.
 */

#include <Kripke/SweepSolver.h>

#include <Kripke.h>
#include <Kripke/Kernel.h>
#include <Kripke/ParallelComm.h>
#include <Kripke/Timing.h>
#include <Kripke/VarTypes.h>
#include <vector>
#include <stdio.h>

struct mpi_request_range {
  mpi_request_range(int n) : requests(n, MPI_REQUEST_NULL) {}
  mpi_request_range(std::vector<MPI_Request> const &v) : requests(v) {}

  std::vector<MPI_Request> requests;
  size_t done = 0;
  int ridx = -1;

  struct end_tag {
  };
  struct request_iter {
  private:
    void check_first()
    {
      // ensure populated on first deref
      if (range.done < 0) {
        this->operator++();
      }
    }

  public:
    request_iter(mpi_request_range &ran) : range(ran) {}
    request_iter &operator++()
    {
      range.wait_next();
      return *this;
    }
    std::tuple<MPI_Request *, int> operator*()
    {
      return {&range.requests[range.ridx], range.ridx};
    }
    MPI_Request *operator->() { return &range.requests[range.ridx]; }
    bool operator==(const end_tag &)
    {
      return range.done == range.requests.size() + 1;
    }
    bool operator!=(const end_tag &)
    {
      return !(range.done == range.requests.size() + 1);
    }
    bool operator==(const request_iter &r) { return &range == &r.range; }
    bool operator!=(const request_iter &r) { return !(*this == r); }

    mpi_request_range &range;
  };
  void wait_next()
  {
    if (done > requests.size()) throw;
    // if (idx > 0) requests[idx] = MPI_REQUEST_NULL;
    MPI_Status s;
    MPI_Waitany(requests.size(), requests.data(), &ridx, &s);
    done++;
  }
  int wait_some(RAJA::impl::Span<int *, int> indices,
                RAJA::impl::Span<MPI_Status *, int> statuses)
  {
    if (done > requests.size()) throw;
    // if (idx > 0) requests[idx] = MPI_REQUEST_NULL;
    int outcount = indices.size();
    MPI_Waitsome(requests.size(),
                 requests.data(),
                 &outcount,
                 indices.data(),
                 statuses.data());
    done += outcount;
    return outcount;
  }
  int try_some(RAJA::impl::Span<int *, int> indices,
               RAJA::impl::Span<MPI_Status *, int> statuses)
  {
    if (done > requests.size()) throw;
    // if (idx > 0) requests[idx] = MPI_REQUEST_NULL;
    int outcount = indices.size();
    MPI_Testsome(requests.size(),
                requests.data(),
                &outcount,
                indices.data(),
                statuses.data());
    done += outcount;
    return outcount;
  }

  size_t size() { return requests.size(); }
  request_iter begin()
  {
    if (ridx < 0) {
      wait_next();
    }
    return request_iter(*this);
  }
  end_tag end() { return {}; }
};

template <typename Range, typename Body>
mpi_request_range make_request_iter(Range &&r, Body &&b)
{
  using std::distance;
  auto dist = distance(r.begin(), r.end());

  using std::begin;
  auto bit = begin(r);
  mpi_request_range requests(dist);
  for (int i = 0; i < dist; ++i) {
    b(bit[i], requests.requests[i]);
  }
  return requests;
}

template <typename InputRange, typename Body>
void forall_input(InputRange &&it, Body &&b)
{
#pragma omp parallel
#pragma omp master
  {
    for (auto const &r : it) {
#pragma omp task
      {
        b(r);
      }
    }
  }
}

template <typename InputRange, typename Body, typename Body2>
void forall_input_funnel(InputRange &&it, Body &&b, Body2 &&funnel_body)
{
  // queue of return type of b
  using ret_type = typename std::result_of<Body(decltype(*it.begin()))>::type;
  tbb::concurrent_queue<ret_type> q;
#pragma omp parallel
#pragma omp master
  {
    for (auto const &r : it) {
#pragma omp task
      {
        q.emplace(b(r));
      }
      ret_type v;
      // must process here, just in case we're serialized
      while (q.try_pop(v)) {

#pragma omp critical
        std::cout << "got data to send: " << v << std::endl;
        funnel_body(v);
// do everything we can to ensure deadlock avoidance
#pragma omp taskyield
      }
    }
  }
  // all receives are done at this point
  while (!q.empty()) {
    std::cout << "draining queue " << q.unsafe_size() << std::endl;
    ret_type v;
    if (q.try_pop(v)) {
      std::cout << "got data to send: " << v << std::endl;
      funnel_body(v);
    }
  }
}

template <typename InputRange, typename Body, typename Body2>
void forall_mpi_progress(InputRange &&it, Body &&b, Body2 &&funnel_body)
{
  // queue of return type of b
  using ret_type = typename std::result_of<Body(decltype(*it.begin()))>::type;
  tbb::concurrent_queue<ret_type> q;
#pragma omp parallel
#pragma omp master
  {
    int indices[10];
    MPI_Status statuses[10];
    while (it.done < it.size()) {
      int cnt = it.try_some(indices, statuses);
      for (int i = 0; i < cnt; ++i) {
#pragma omp task
        {
          int &idx = indices[i];
          q.emplace(b({&it.requests[idx], idx}));
        }
      }
      ret_type v;
      // must process here, just in case we're serialized
      while (q.try_pop(v)) {

#pragma omp critical
        std::cout << "got data to send: " << v << std::endl;
        funnel_body(v);
        // do everything we can to ensure deadlock avoidance
#pragma omp taskyield
      }
    }
  }
  // all receives are done at this point
  while (!q.empty()) {
    std::cout << "draining queue " << q.unsafe_size() << std::endl;
    ret_type v;
    if (q.try_pop(v)) {
      std::cout << "got data to send: " << v << std::endl;
      funnel_body(v);
    }
  }
}

using namespace Kripke;

/**
  Perform full parallel sweep algorithm on subset of subdomains.
*/
void Kripke::SweepSolver (Kripke::Core::DataStore &data_store, std::vector<SdomId> subdomain_list, bool block_jacobi)
{
  KRIPKE_TIMER(data_store, SweepSolver);

  // Initialize plane data
  Kripke::Kernel::kConst(data_store.getVariable<Field_IPlane>("i_plane"), 0.0);
  Kripke::Kernel::kConst(data_store.getVariable<Field_JPlane>("j_plane"), 0.0);
  Kripke::Kernel::kConst(data_store.getVariable<Field_KPlane>("k_plane"), 0.0);

  // Create a new sweep communicator object
  SweepComm *comm = comm = new SweepComm(data_store);

  // Add all subdomains in our list
  for(size_t i = 0;i < subdomain_list.size();++ i){
//    Kripke::Core::Comm default_comm;
//    printf("SweepSolver: rank=%d, sdom=%d\n", (int)default_comm.rank(), (int)*subdomain_list[i]);
    SdomId sdom_id = subdomain_list[i];
    comm->addSubdomain(data_store, sdom_id);
  }

  auto &field_upwind = data_store.getVariable<Field_Adjacency>("upwind");

  /* Loop until we have finished all of our work */
#pragma omp parallel
#pragma omp master
  while(comm->workRemaining()){

    std::vector<SdomId> sdom_ready = comm->readySubdomains();
    int backlog = sdom_ready.size();

    // Run top of list
    if(backlog > 0){
#pragma omp task
      {
        SdomId sdom_id = sdom_ready[0];

        auto upwind = field_upwind.getView(sdom_id);

        // Clear boundary conditions
        if(upwind(Direction{0}) == -1){
          Kripke::Kernel::kConst(data_store.getVariable<Field_IPlane>("i_plane"), sdom_id, 0.0);
        }
        if(upwind(Direction{1}) == -1){
          Kripke::Kernel::kConst(data_store.getVariable<Field_JPlane>("j_plane"), sdom_id, 0.0);
        }
        if(upwind(Direction{2}) == -1){
          Kripke::Kernel::kConst(data_store.getVariable<Field_KPlane>("k_plane"), sdom_id, 0.0);
        }

        // Perform subdomain sweep
        Kripke::Kernel::sweepSubdomain(data_store, Kripke::SdomId{sdom_id});

        // Mark as complete (and do any communication)
#pragma omp critical
        comm->markComplete(sdom_id);
      }
    }
  }

  delete comm;

//  printf("\nAfter sweep psi:\n");
//  data_store.getVariable<Field_Flux>("psi").dump();

}


