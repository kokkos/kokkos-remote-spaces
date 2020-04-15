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

#include "Context.hpp"

#include <quo.h>
#include <stdlib.h>

#include "handle_error.hpp"

namespace quo {
namespace {
QUO_obj_type_t map_to_quo(ObjectType type) {
  switch (type) {
  case ObjectType::MACHINE:
    return QUO_OBJ_MACHINE;
  case ObjectType::NODE:
    return QUO_OBJ_NODE;
  case ObjectType::SOCKET:
    return QUO_OBJ_SOCKET;
  case ObjectType::CORE:
    return QUO_OBJ_CORE;
  case ObjectType::PROCESSING_UNIT:
    return QUO_OBJ_PU;
  }
}

QUO_bind_push_policy_t map_to_quo(BindPushPolicy policy) {
  switch (policy) {
  case BindPushPolicy::PROVIDED:
    return QUO_BIND_PUSH_PROVIDED;
  case BindPushPolicy::OBJECT:
    return QUO_BIND_PUSH_OBJ;
  }
}
}

struct Context::Impl {
  QUO_context ctx;
};

Context::Context(MPI_Comm comm) : m_impl(new Impl()) {
  QUO_CXX_HANDLE_ERROR(QUO_create(&(m_impl->ctx), comm));
}

Context::~Context() { QUO_free(m_impl->ctx); }

int Context::nobjs_by_type(ObjectType type) const {
  int nobjs{0};

  QUO_CXX_HANDLE_ERROR(
      QUO_nobjs_by_type(m_impl->ctx, map_to_quo(type), &nobjs));

  return nobjs;
}

int Context::nobjs_in_type_by_type(ObjectType in_type, int index,
                                   ObjectType type) const {
  int nobjs{0};

  QUO_CXX_HANDLE_ERROR(QUO_nobjs_in_type_by_type(
      m_impl->ctx, map_to_quo(in_type), index, map_to_quo(type), &nobjs));

  return nobjs;
}

bool Context::cpuset_in_type(ObjectType type, int index) const {
  int in_set{0};

  QUO_cpuset_in_type(m_impl->ctx, map_to_quo(type), index, &in_set);

  return (in_set != 0);
}

std::vector<int> Context::qids_in_type(ObjectType type, int index) const {
  int n_quids, *p;

  QUO_CXX_HANDLE_ERROR(
      QUO_qids_in_type(m_impl->ctx, map_to_quo(type), index, &n_quids, &p));

  /* Make sure p is free'ed. */
  auto sp = std::shared_ptr<int>(p, [](int *p) { free(p); });

  /* Copy is deliberate, for lack of a better alternative. */
  return std::vector<int>(p, p + n_quids);
}

int Context::nnumanodes() const {
  int nnumanodes{0};

  QUO_CXX_HANDLE_ERROR(QUO_nnumanodes(m_impl->ctx, &nnumanodes));

  return nnumanodes;
}

int Context::nsockets() const {
  int nsockets{0};

  QUO_CXX_HANDLE_ERROR(QUO_nsockets(m_impl->ctx, &nsockets));

  return nsockets;
}

int Context::ncores() const {
  int ncores{0};

  QUO_CXX_HANDLE_ERROR(QUO_ncores(m_impl->ctx, &ncores));

  return ncores;
}

int Context::nnodes() const {
  int nnodes{0};

  QUO_CXX_HANDLE_ERROR(QUO_nnodes(m_impl->ctx, &nnodes));

  return nnodes;
}

int Context::nqids() const {
  int nqids{0};

  QUO_CXX_HANDLE_ERROR(QUO_nqids(m_impl->ctx, &nqids));

  return nqids;
}

int Context::id() const {
  int id{0};

  QUO_CXX_HANDLE_ERROR(QUO_id(m_impl->ctx, &id));

  return id;
}

bool Context::bound() const {
  int bound{0};

  QUO_CXX_HANDLE_ERROR(QUO_bound(m_impl->ctx, &bound));

  return (bound != 0);
}

std::string Context::stringify_cbind() const {
  char *buffer{nullptr};

  QUO_CXX_HANDLE_ERROR(QUO_stringify_cbind(m_impl->ctx, &buffer));

  /* Make sure buffer is free'ed. */
  auto sp = std::shared_ptr<char>(buffer, [](char *p) { free(p); });

  /* Copy is deliberate, for lack of a better alternative. */
  return std::string(buffer);
}

void Context::bind_push(BindPushPolicy policy, ObjectType type,
                        int index) const {
  QUO_CXX_HANDLE_ERROR(
      QUO_bind_push(m_impl->ctx, map_to_quo(policy), map_to_quo(type), index));
}

void Context::bind_pop() const {
  QUO_CXX_HANDLE_ERROR(QUO_bind_pop(m_impl->ctx));
}

bool Context::auto_distrib(ObjectType distrib_over_this,
                           int max_qids_per_res_type) const {
  int selected{0};

  QUO_CXX_HANDLE_ERROR(QUO_auto_distrib(m_impl->ctx,
                                        map_to_quo(distrib_over_this),
                                        max_qids_per_res_type, &selected));

  return (selected != 0);
}

} /* namespace quo */
