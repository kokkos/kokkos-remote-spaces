#include<GenerateMatrix.hpp>

template<class YType,class AType,class XType>
void spmv(YType y, AType A, XType x) {
#ifdef KOKKOS_ENABLE_CUDA
  int rows_per_team = 64;
  int team_size = 64;
#else
  int rows_per_team = 512;
  int team_size = 1;
#endif
  int64_t nrows = y.extent(0);
  Kokkos::parallel_for("SPMV", Kokkos::TeamPolicy<>
       ((nrows+rows_per_team-1)/rows_per_team,team_size,8),
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
        [=] (const int64_t i,double& sum) {
        sum += A.values(i+row_start)*x(A.col_idx(i+row_start));
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


template<class VType,class AType>
void cg_solve(VType y, AType A, VType b) {
  int max_iter = 200;
  int myproc = 0;
  double tolerance = 1e-6;
  int num_iters = 0;

  double normr = 0;
  double rtrans = 0;
  double oldrtrans = 0;

  int64_t print_freq = max_iter/10;
  if (print_freq>50) print_freq = 50;
  if (print_freq<1)  print_freq = 1;
  VType x("x",b.extent(0));
  VType r("r",x.extent(0));
  VType p("r",x.extent(0)); // Needs to be global
  VType Ap("r",x.extent(0));
  double one = 1.0;
  double zero = 0.0;

  axpby(p, one, x, zero, x);

  spmv(Ap, A, p);

  axpby(r, one, b, -one, Ap);

  rtrans = dot(r, r);

  normr = std::sqrt(rtrans);

  if (myproc == 0) {
    std::cout << "Initial Residual = "<< normr << std::endl;
  }

  double brkdown_tol = std::numeric_limits<double>::epsilon();

  for(int64_t k=1; k <= max_iter && normr > tolerance; ++k) {
    if (k == 1) {
      axpby(p, one, r, zero, r);
    }
    else {
      oldrtrans = rtrans;
      rtrans = dot(r, r);
      double beta = rtrans/oldrtrans;
      axpby(p, one, r, beta, p);
    }

    normr = std::sqrt(rtrans);

    if (myproc == 0 && (k%print_freq==0 || k==max_iter)) {
      std::cout << "Iteration = "<<k<<"   Residual = "<<normr<<std::endl;
    }

    double alpha = 0;
    double p_ap_dot = 0;

    spmv(Ap, A, p);

    p_ap_dot = dot(Ap, p);

    if (p_ap_dot < brkdown_tol) {
      if (p_ap_dot < 0 ) {
        std::cerr << "miniFE::cg_solve ERROR, numerical breakdown!"<<std::endl;
        return;
      }
      else brkdown_tol = 0.1 * p_ap_dot;
    }
    alpha = rtrans/p_ap_dot;

    axpby(x, one, x, alpha, p);
    axpby(r, one, r, -alpha, Ap);
    num_iters = k;
  }

}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc,argv);
  {
    int N = atoi(argv[1]);
    CrsMatrix<Kokkos::HostSpace> A = Impl::generate_miniFE_matrix(N);
    Kokkos::View<double*,Kokkos::HostSpace> x = Impl::generate_miniFE_vector(N);
    Kokkos::View<double*,Kokkos::HostSpace> y("Y",x.extent(0));
    /*for(int i=0; i<N*N*N; i++) {
      printf("%i ",i);
      for(int j=m.row_ptr(i);j<m.row_ptr(i+1);j++)
        printf("(%i %li,%lf) ",j,m.col_idx(j),m.values(j));
      printf(" = %lf\n",v(i));
    }*/


    cg_solve(y,A,x);

  }
  Kokkos::finalize();
}
