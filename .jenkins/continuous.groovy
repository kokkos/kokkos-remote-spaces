pipeline {
    agent none 

    stages {
        stage('Build') {
            parallel {
                stage('shmem-openmpi') {
                    agent {
                        dockerfile {
                            filename 'Dockerfile.openmpi'
                            dir 'scripts/docker'
                            label 'docker'
                        }
                    }
                    environment {
                        OMP_NUM_THREADS = 8
                        OMP_PROC_BIND = 'true'
                    }
                    steps {
                        sh '''rm -rf build && mkdir -p build && cd build && \
                              cmake \
                                -DCMAKE_BUILD_TYPE=Release \
                                -DKokkos_DIR=${KOKKOS_ROOT} \
                                -DKokkos_ENABLE_SHMEMSPACE=ON \
                                -DSHMEM_ROOT=/opt/openmpi \
                                -DKokkos_ENABLE_TESTS=ON \
                                -DCMAKE_CXX_FLAGS=-Werror \
                              .. && \
                              make -j8 && cd unit_tests && mpirun -np 2 ./KokkosRemote_TestAll'''
                    }
                }
            }
        }
    }
}
