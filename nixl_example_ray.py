#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import ray
import logging

from nixl_example_no_ray import Target, Initiator

logger = logging.getLogger(__name__)

@ray.remote
class TargetActor(Target):

    def name(self):
        return self.name

@ray.remote
class InitiatorActor(Initiator):

    def name(self):
        return self.name


def test_nixl_api_with_ray():
    if not ray.is_initialized():
        ray.init()

    env_vars = {}
    # Still work without this.
    # if os.environ.get("NIXL_PLUGIN_DIR"):
    #     env_vars["NIXL_PLUGIN_DIR"] = os.environ["NIXL_PLUGIN_DIR"]

    remote_options = dict(
        runtime_env=dict(env_vars=env_vars),
        num_gpus=1,
    )

    target = TargetActor.options(**remote_options).remote()
    initiator = InitiatorActor.options(**remote_options).remote()

    target_meta = target.prepare_for_read.remote()

    handle_initiator = initiator.blocking_read_remote.remote(target_meta)

    handle_target = target.blocking_wait_for_read.remote(initiator.name.remote(), b"UUID1")

    ray.get([handle_initiator, handle_target])

    ray.get(target.cleanup.remote())
    ray.get(initiator.cleanup.remote())

    logger.info("Test complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_nixl_api_with_ray()
