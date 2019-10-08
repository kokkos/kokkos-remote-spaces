#include<Kokkos_RemoteSpaces.hpp>
#include<GenerateMatrix.hpp>
typedef Kokkos::DefaultRemoteMemorySpace remote_space;
typedef Kokkos::View<double**, Kokkos::DefaultRemoteMemorySpace> remote_view_type;

int rows_per_team_set = 0;
int team_size_set = 0;
int vector_length = 8;

template<class YType,class AType,class XType>
void spmv(YType y, AType A, XType x) {
#ifdef KOKKOS_ENABLE_CUDA
  int rows_per_team = rows_per_team_set?rows_per_team_set:16;
  int team_size = team_size_set?team_size_set:16;
#else
  int rows_per_team = 512;
  int team_size = 1;
#endif
  int64_t nrows = y.extent(0);
  auto policy = require(Kokkos::TeamPolicy<>
       ((nrows+rows_per_team-1)/rows_per_team,team_size,vector_length),
    Kokkos::Experimental::WorkItemProperty::HintHeavyWeight);
  Kokkos::parallel_for("SPMV", policy, 
    KOKKOS_LAMBDA (const Kokkos::TeamPolicy<>::member_type& team) {
    const int64_t first_row = team.league_rank()*rows_per_team;
    const int64_t last_row = first_row+rows_per_team<nrows?
                         first_row+rows_per_team:nrows;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team,first_row,last_row),
      [&] (const int64_t row) {
      const int64_t row_start=A.row_ptr(row);
      const int64_t row_length=A.row_ptr(row+1)-row_start;

      double y_row;
      Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team,row_length),
        [&] (const int64_t i,double& sum) {
        int64_t idx = A.col_idx(i+row_start);
        sum += A.values(i+row_start)*x(idx/MASK,idx%MASK);
      },y_row);
      y(row) = y_row;
    });
  });
}

template<class YType, class XType>
double dot(YType y, XType x) {
  double result;
  Kokkos::parallel_reduce("DOT",y.extent(0),KOKKOS_LAMBDA (const int64_t& i, double& lsum) {
    lsum += y(i)*x(i);
  },result);
  return result;
}

template<class ZType, class YType, class XType>
void axpby(ZType z, double alpha, XType x, double beta,  YType y) {
  int64_t n = z.extent(0);
  Kokkos::parallel_for("AXPBY", n, KOKKOS_LAMBDA ( const int& i) {
    z(i) = alpha*x(i) + beta*y(i);
  });
}

template<class VType> 
void print_vector(int label, VType v) {
  printf("\n\nPRINT %s\n\n",v.label().c_str());
  
  int myRank,numRanks;
  MPI_Comm_rank(MPI_COMM_WORLD,&myRank);
  MPI_Comm_size(MPI_COMM_WORLD,&numRanks);
  MPI_Barrier(MPI_COMM_WORLD);
  for(int r = 0; r<numRanks; r++) {
    if(r==myRank)
    Kokkos::parallel_for(v.extent(0), KOKKOS_LAMBDA(const int i) {
      printf("%i %i %i %lf\n",label,myRank,i,v(i));
    });
    Kokkos::fence();
    MPI_Barrier(MPI_COMM_WORLD);
  }
  printf("\n\nPRINT DONE: %s\n\n",v.label().c_str());
}

template<class VType,class AType, class PType>
int cg_solve(VType y, AType A, VType b, PType p_global,int max_iter, double tolerance) {
  int myproc = 0;
  int num_iters = 0;

  double normr = 0;
  double rtrans = 0;
  double oldrtrans = 0;

  int64_t print_freq = max_iter/10;
  if (print_freq>50) print_freq = 50;
  if (print_freq<1)  print_freq = 1;
  VType x("x",b.extent(0));
  VType r("r",x.extent(0));
  VType p(p_global.data(),x.extent(0)); // Needs to be global
  VType Ap("r",x.extent(0));
  double one = 1.0;
  double zero = 0.0;

  axpby(p, one, x, zero, x);
  remote_space().fence();
  spmv(Ap, A, p_global);
  remote_space().fence();
  //print_vector(300000000,Ap);
  //print_vector(400000000,p);
  //print_vector(500000000,b);

  axpby(r, one, b, -one, Ap);

  rtrans = dot(r, r);
  MPI_Allreduce(MPI_IN_PLACE,&rtrans,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

  normr = std::sqrt(rtrans);

  if (myproc == 0) {
    std::cout << "Initial Residual = "<< normr << "( " << myproc << " " << " " << x.extent(0) << " )" << std::endl;
  }

  double brkdown_tol = std::numeric_limits<double>::epsilon();
  Kokkos::Timer timer;
  for(int64_t k=1; k <= max_iter && normr > tolerance; ++k) {
    if (k == 1) {
      axpby(p, one, r, zero, r);
      remote_space().fence();
    }
    else {
      oldrtrans = rtrans;
      rtrans = dot(r, r);
      MPI_Allreduce(MPI_IN_PLACE,&rtrans,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
      double beta = rtrans/oldrtrans;
      axpby(p, one, r, beta, p);
      remote_space().fence();
    }
//  print_vector(100000000+k,p);
    normr = std::sqrt(rtrans);

    if (myproc == 0 && (k%print_freq==0 || k==max_iter)) {
      std::cout << "Iteration = "<<k<<"   Residual = "<<normr<< " " << rtrans/oldrtrans << std::endl;
    }

    double alpha = 0;
    double p_ap_dot = 0;

    spmv(Ap, A, p_global);
//  print_vector(200000000+k,Ap);

    p_ap_dot = dot(Ap, p);
    //p_ap_dot = dot(p, p);
    MPI_Allreduce(MPI_IN_PLACE,&p_ap_dot,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    
    if (p_ap_dot < brkdown_tol) {
      if (p_ap_dot < 0 ) {
        std::cerr << "miniFE::cg_solve ERROR, numerical breakdown!"<<std::endl;
        return num_iters;
      }
      else brkdown_tol = 0.1 * p_ap_dot;
    }
    alpha = rtrans/p_ap_dot;

    axpby(x, one, x, alpha, p);
    axpby(r, one, r, -alpha, Ap);
    num_iters = k;
  }
  double time = timer.seconds();
  printf("Seconds: %lf\n",time);
  return num_iters;
}

int main(int argc, char* argv[]) {

  MPI_Init(&argc,&argv);
  #ifdef KOKKOS_ENABLE_SHMEMSPACE
  shmem_init();
  #endif
  #ifdef KOKKOS_ENABLE_NVSHMEMSPACE
  MPI_Comm mpi_comm;
  nvshmemx_init_attr_t attr;
  mpi_comm = MPI_COMM_WORLD;
  attr.mpi_comm = &mpi_comm;
  nvshmemx_init_attr (NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
  #endif

  int myRank,numRanks;
  MPI_Comm_rank(MPI_COMM_WORLD,&myRank);
  MPI_Comm_size(MPI_COMM_WORLD,&numRanks);

  Kokkos::initialize(argc,argv);


  {
    int N = argc>1?atoi(argv[1]):100;
    int max_iter = argc>2?atoi(argv[2]):200;
    double tolerance = argc>3?atoi(argv[3]):1e-7;
    if(argc>4)
      rows_per_team_set = atoi(argv[4]);
    if(argc>5)
      team_size_set = atoi(argv[5]);
    if(argc>6)
      vector_length = atoi(argv[6]);

    CrsMatrix<Kokkos::HostSpace> h_A = Impl::generate_miniFE_matrix(N);
    Kokkos::View<double*,Kokkos::HostSpace> h_x = Impl::generate_miniFE_vector(N);

    Kokkos::View<int64_t*> row_ptr("row_ptr",h_A.row_ptr.extent(0));
    Kokkos::View<LOCAL_ORDINAL*> col_idx("col_idx",h_A.col_idx.extent(0));
    Kokkos::View<double*> values("values",h_A.values.extent(0));
    CrsMatrix<Kokkos::DefaultExecutionSpace::memory_space> A(row_ptr,col_idx,values,h_A.num_cols());
    Kokkos::View<double*> x("X",h_x.extent(0));
    Kokkos::View<double*> y("Y",h_x.extent(0));

    int* rank_list = new int[numRanks];
    for(int r=0; r<numRanks; r++)
      rank_list[r] = r; 
    remote_view_type p =
      Kokkos::allocate_symmetric_remote_view<remote_view_type>("MyView",numRanks,rank_list,(h_x.extent(0)+numRanks-1)/numRanks);
    /*Kokkos::parallel_for(1, KOKKOS_LAMBDA(const int) {
      a(0,0,0) = 0;
    });*/

    Kokkos::deep_copy(x,h_x);
    Kokkos::deep_copy(A.row_ptr,h_A.row_ptr);
    Kokkos::deep_copy(A.col_idx,h_A.col_idx);
    Kokkos::deep_copy(A.values,h_A.values);

    if(false)
    for(int r=0; r<numRanks;r++) {
    if(r==myRank) {
    printf("Extents: %i %i %i\n",r,h_A.nnz(),h_x.extent(0));

    for(int i=0; i<h_A.num_rows(); i++) {
      printf("%i ",i);
      for(int j=h_A.row_ptr(i);j<h_A.row_ptr(i+1);j++)
        //printf("(%i , %li , %lf) ",j,h_A.col_idx(j),h_A.values(j));
        printf("(%i , %li) ",j,h_A.col_idx(j));
      printf(" = %lf\n",h_x(i));
    }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    }

    int64_t start_row = myRank * p.extent(1);
    int64_t end_row = (myRank + 1)* p.extent(1);
    if(end_row > h_x.extent(0)) end_row = h_x.extent(0);
    Kokkos::pair<int64_t,int64_t> bounds(start_row,end_row);
    Kokkos::Timer timer;
    
    int num_iters = cg_solve(y,A,Kokkos::View<double*>(Kokkos::subview(x,bounds)),p,max_iter,tolerance);
    double time = timer.seconds();

    // Compute Bytes and Flops
    double spmv_bytes = A.num_rows() * sizeof(int64_t) + A.nnz() * sizeof(LOCAL_ORDINAL) + A.nnz() * sizeof(double) + 
                        A.nnz() * sizeof(double) + A.num_rows() * sizeof(double);

    double dot_bytes = x.extent(0) * sizeof(double) * 2;
    double axpby_bytes = x.extent(0) * sizeof(double) * 3;

    double spmv_flops = A.nnz() * 2;
    double dot_flops = x.extent(0) * 2;
    double axpby_flops = x.extent(0) * 3;

    int spmv_calls = 1 + num_iters;
    int dot_calls = num_iters;
    int axpby_calls = 2 + num_iters * 3; 

    double GFlops = 1e-9 * (spmv_flops*spmv_calls + dot_flops*dot_calls + axpby_flops*axpby_calls)/time;
    double GBs = (1.0/1024/1024/1024) * (spmv_bytes*spmv_calls + dot_bytes*dot_calls + axpby_bytes*axpby_calls)/time;

    MPI_Allreduce(MPI_IN_PLACE,&GFlops,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&GBs,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    if(myRank == 0) {
      printf("CGSolve for 3D (%i %i %i); %i iterations; %li nnz/rank; %lf time\n",N,N,N,num_iters,A.nnz(),time);
      printf("Performance: %lf GFlop/s %lf GB/s (Calls SPMV: %i Dot: %i AXPBY: %i\n",
        GFlops, GBs , spmv_calls, dot_calls, axpby_calls);
    }

  }
  Kokkos::finalize();

  MPI_Finalize();
}
