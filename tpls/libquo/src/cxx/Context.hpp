/**
 * Copyright (c) 2016 Los Alamos National Security, LLC
 *                         All rights reserved.
 *
 * This software was produced under U.S. Government contract DE-AC52-06NA25396
 * for Los Alamos National Laboratory (LANL), which is operated by Los Alamos
 * National Security, LLC for the U.S. Department of Energy. The U.S. Government
 * has rights to use, reproduce, and distribute this software.  NEITHER THE
 * GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS
 * OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If
 * software is modified to produce derivative works, such modified software
 * should be clearly marked, so as not to confuse it with the version available
 * from LANL.
 *
 * Additionally, redistribution and use in source and binary forms, with or
 * without modification, are permitted provided that the following conditions
 * are met:
 *
 * · Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * · Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * · Neither the name of Los Alamos National Security, LLC, Los Alamos
 *   National Laboratory, LANL, the U.S. Government, nor the names of its
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
 * NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL
 * SECURITY, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef QUO_CXX_CONTEXT_HPP
#define QUO_CXX_CONTEXT_HPP

#include <memory>
#include <string>
#include <vector>

#include "mpi.h"

#include "types.hpp"

namespace quo {

/**
 * @brief Wrapper class for a libquo context.
 *
 * This class provides a direct wrapper for a libquo context
 * and the functions that operate on it.
 * Names and signatures are the same as in libquo,
 * modulo the context parameter and value semantics.
 * For detailed documentation see the corresponding
 * libquo documentation
 */
class Context {
public:
  explicit Context(MPI_Comm comm = MPI_COMM_WORLD);
  ~Context();

  /**
   * @brief Numer of objects of a specific type.
   *
   * Returns the number of objects for the calling node.
   */
  int nobjs_by_type(ObjectType type) const;

  /**
   * @brief Number of objects of a specific type on a specific instance.
   *
   * Returns the number of objects for the object index.
   */
  int nobjs_in_type_by_type(ObjectType in_type, int index,
                            ObjectType type) const;

  /**
   * @brief Is the calling node in index-th of type.
   */
  bool cpuset_in_type(ObjectType type, int index) const;

  /**
   * @brief Number of qid in a specific object.
   */
  std::vector<int> qids_in_type(ObjectType type, int index) const;

  /**
   * @brief Number of NUMA nodes on the system of the caller.
   */
  int nnumanodes() const;

  /**
   * @brief Number of sockets on the system of the caller.
   */
  int nsockets() const;

  /**
   * @brief Number of cores on the system of the caller.
   */
  int ncores() const;

  /**
   * @brief Number of machines the caller is bound to.
   */
  int nnodes() const;

  /**
   * @brief Number of qids on this machine.
   */
  int nqids() const;

  /**
   * @brief qid of the caller.
   */
  int id() const;

  bool bound() const;

  /**
   * @brief String representation of the binding of the caller
   */
  std::string stringify_cbind() const;

  /**
   * @brief Set new binding.
   */
  void bind_push(BindPushPolicy policy, ObjectType type, int index) const;

  /**
   * @brief Return to last binding.
   */
  void bind_pop() const;

  /**
   * @brief Local barrier.
   */
  void barrier() const;

  /**
   * @brief Automatically select qids on ressource.
   *
   * @return True if the caller is selected.
   */
  bool auto_distrib(ObjectType distrib_over_this,
                    int max_qids_per_res_type) const;

private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;
};

} /* namespace quo */

#endif
