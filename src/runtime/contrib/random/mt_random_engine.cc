/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file random/mt_random_engine.cc
 * \brief mt19937 random engine
 */
#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/threading_backend.h>

#include <algorithm>
#include <ctime>
#include <random>
#include <thread>

#include "../3rdparty/compiler-rt/builtin_fp16.h"

namespace tvm {
namespace contrib {

/*!
 * \brief An interface for generating [tensors of] random numbers.
 */
class RandomEngine {
 public:
  /*!
   * \brief Creates a RandomEngine using a default seed.
   */
  RandomEngine() { this->Seed(time(nullptr)); }

  /*!
   * \brief Creates a RandomEngine, suggesting the use of a provided seed.
   */
  explicit RandomEngine(unsigned seed) { this->Seed(seed); }

  /*!
   * \brief Seeds the underlying RNG, if possible.
   */
  inline void Seed(unsigned seed) {
    rnd_engine_.seed(seed);
    this->rseed_ = static_cast<unsigned>(seed);
  }

  /*!
   * \return the seed associated with the underlying RNG.
   */
  inline unsigned GetSeed() const { return rseed_; }

  /*!
   * \return a random integer sampled from the RNG.
   */
  inline unsigned GetRandInt() { return rnd_engine_(); }

  /*!
   * \brief Fills a tensor with values drawn from Unif(low, high)
   */
  void SampleUniform(DLTensor* data, float low, float high) {
    ICHECK_GT(high, low) << "high must be bigger than low";
    ICHECK(data->strides == nullptr);

    DLDataType dtype = data->dtype;
    int64_t size = 1;
    for (int i = 0; i < data->ndim; ++i) {
      size *= data->shape[i];
    }

    ICHECK(dtype.code == kDLFloat && dtype.bits == 32 && dtype.lanes == 1);

    if (data->device.device_type == kDLCPU) {
      std::uniform_real_distribution<float> uniform_dist(low, high);
      std::generate_n(static_cast<float*>(data->data), size,
                      [&]() { return uniform_dist(rnd_engine_); });
    } else {
      LOG(FATAL) << "Do not support random.uniform on this device yet";
    }
  }

  /*!
   * \brief Fills a tensor with values drawn from Normal(loc, scale**2)
   */
  void SampleNormal(DLTensor* data, float loc, float scale) {
    ICHECK_GT(scale, 0) << "standard deviation must be positive";
    ICHECK(data->strides == nullptr);

    DLDataType dtype = data->dtype;
    int64_t size = 1;
    for (int i = 0; i < data->ndim; ++i) {
      size *= data->shape[i];
    }

    ICHECK(dtype.code == kDLFloat && dtype.bits == 32 && dtype.lanes == 1);

    if (data->device.device_type == kDLCPU) {
      std::normal_distribution<float> normal_dist(loc, scale);
      std::generate_n(static_cast<float*>(data->data), size,
                      [&]() { return normal_dist(rnd_engine_); });
    } else {
      LOG(FATAL) << "Do not support random.normal on this device yet";
    }
  }

  void RandomFill(DLTensor* data) {
    if (data->device.device_type == kDLCPU) {
      FillData(data);
    } else {
      runtime::NDArray local = runtime::NDArray::Empty(
          std::vector<int64_t>{data->shape, data->shape + data->ndim}, data->dtype, {kDLCPU, 0});
      DLTensor* tensor = const_cast<ffi::NDArrayObj*>(local.operator->());
      FillData(tensor);
      runtime::NDArray::CopyFromTo(tensor, data);
    }
  }

  void RandomFillForMeasure(DLTensor* data) {
    if (data->device.device_type == kDLCPU) {
      FillDataForMeasure(data);
    } else {
      runtime::NDArray local = runtime::NDArray::Empty(
          std::vector<int64_t>{data->shape, data->shape + data->ndim}, data->dtype, {kDLCPU, 0});
      DLTensor* tensor = const_cast<ffi::NDArrayObj*>(local.operator->());
      FillDataForMeasure(tensor);
      runtime::NDArray::CopyFromTo(tensor, data);
    }
  }

 private:
  void FillDataImpl(void* data, int64_t st, int64_t ed, DLDataType dtype) {
    // Make the value be 1.0 - 10.0, not (0.0 - 1.0) so that we could satisfy
    // quantized dtype (uint8 / int8) data non-empty requirement
    std::uniform_real_distribution<> dist(1.0, 10.0);
    // Use float representation could make us work well on float / int type too.
    if (dtype.bits == 1) {
      std::generate_n(static_cast<bool*>(data) + st, ed - st, [&]() { return dist(rnd_engine_); });
    } else if (dtype.bits == 4) {
      // For uint4/int4 we pack two values into a single byte.
      // Thus, to ensure both values are non-zero, we use a distribution of 17 - 30.
      std::uniform_real_distribution<> packed_dist(17.0, 30.0);
      std::generate_n(reinterpret_cast<uint8_t*>(data) + st, ed - st,
                      [&]() { return packed_dist(rnd_engine_); });
    } else if (dtype.bits == 8) {
      std::generate_n(static_cast<uint8_t*>(data) + st, ed - st,
                      [&]() { return dist(rnd_engine_); });
    } else if (dtype.bits == 16) {
      std::generate_n(static_cast<uint16_t*>(data) + st, ed - st, [&]() {
        return __truncXfYf2__<float, uint32_t, 23, uint16_t, uint16_t, 10>(
            static_cast<float>(dist(rnd_engine_)));
      });
    } else if (dtype.bits == 32) {
      std::generate_n(static_cast<float*>(data) + st, ed - st, [&]() { return dist(rnd_engine_); });
    } else if (dtype.bits == 64) {
      std::generate_n(static_cast<double*>(data) + st, ed - st,
                      [&]() { return dist(rnd_engine_); });
    } else {
      LOG(FATAL) << "Doesn't support dtype code " << dtype.code << " dtype bits " << dtype.bits;
    }
  }

  void FillData(DLTensor* tensor) {
    int64_t size = 1;
    for (int i = 0; i < tensor->ndim; ++i) {
      size *= tensor->shape[i];
    }
    DLDataType dtype = tensor->dtype;
    if (dtype.bits == 1 || dtype.bits == 4 || dtype.bits == 8 || dtype.bits == 16 ||
        dtype.bits == 32 || dtype.bits == 64) {
      FillDataImpl(tensor->data, 0, size, dtype);
    } else {
      LOG(FATAL) << "Doesn't support dtype code " << dtype.code << " dtype bits " << dtype.bits;
    }
  }

  void FillDataForMeasure(DLTensor* tensor) {
    struct ParallelTask {
      static int RunTask(int task_id, TVMParallelGroupEnv* penv, void* cdata) {
        ParallelTask* task = static_cast<ParallelTask*>(cdata);
        task->Run(task_id, penv->num_task);
        return 0;
      }

      void Run(int i, int num_tasks) {
        int64_t chunk_size = size / num_tasks;
        int64_t st = i * chunk_size;
        int64_t ed = std::min(st + chunk_size, size);
        self->FillDataImpl(data, st, ed, dtype);
      }

      RandomEngine* self;
      void* data;
      int64_t size;
      DLDataType dtype;
    };

    ParallelTask task;
    task.self = this;
    task.data = tensor->data;
    DLDataType dtype = task.dtype = tensor->dtype;
    int64_t& size = task.size = 1;
    for (int i = 0; i < tensor->ndim; ++i) {
      size *= tensor->shape[i];
    }
    if (dtype.bits == 1 || dtype.bits == 4 || dtype.bits == 8 || dtype.bits == 16 ||
        dtype.bits == 32 || dtype.bits == 64) {
      int res = TVMBackendParallelLaunch(ParallelTask::RunTask, &task, 0);
      ICHECK_EQ(res, 0) << "RandomFillForMeasure: TVMBackendParallelLaunch failed";
    } else {
      LOG(FATAL) << "Doesn't support dtype code " << dtype.code << " dtype bits " << dtype.bits;
    }
  }

 private:
  std::mt19937 rnd_engine_;
  unsigned rseed_;
};

}  // namespace contrib
}  // namespace tvm
