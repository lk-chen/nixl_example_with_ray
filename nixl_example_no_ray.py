#!/usr/bin/env python3

# Adapted from nixl/examples/python/nixl_api_example.py

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
import logging
import time
import torch
from typing import Any
from dataclasses import dataclass

import nixl._utils as nixl_utils
from nixl._api import nixl_agent, nixl_agent_config
from nixl._bindings import nixlXferDList

logger = logging.getLogger("ray")

# True to test DRAM(cpu), False to test tensors(gpu)
TEST_DRAM=False

buf_size = 32

@dataclass
class AggregatedAgentMetadata:
    """Aggregated metadata from NIXL agents"""

    engine_id: str
    agent_metadata: bytes
    xfer_descs: nixlXferDList

class NixlWrapper:
    def __init__(self, name: str, **kwargs):
        logger.info(f"Using NIXL Plugins from: {os.environ['NIXL_PLUGIN_DIR']}")

        self.name = name
        self.nixl_agent = nixl_agent(name, nixl_agent_config(backends=["UCX"]))
        plugin_list = self.nixl_agent.get_plugin_list()
        assert "UCX" in plugin_list

        logger.info("Plugin parameters")
        logger.info(self.nixl_agent.get_plugin_mem_types("UCX"))
        logger.info(self.nixl_agent.get_plugin_params("UCX"))

        logger.info("Loaded backend parameters")
        logger.info(self.nixl_agent.get_backend_mem_types("UCX"))
        logger.info(self.nixl_agent.get_backend_params("UCX"))

        if TEST_DRAM:
            self._addr = nixl_utils.malloc_passthru(buf_size * 2)
            addrs = [(self._addr, buf_size, 0), (self._addr + buf_size, buf_size, 0)]
            strings = [(self._addr, buf_size, 0, "a"), (self._addr + buf_size, buf_size, 0, "b")]
            self._reg_descs = self.nixl_agent.get_reg_descs(strings, "DRAM", is_sorted=True)
            assert self.nixl_agent.register_memory(self._reg_descs) is not None
            self.local_xfer_descs: nixlXferDList = self.nixl_agent.get_xfer_descs(addrs, "DRAM", is_sorted=True)
        else:
            self._tensors = [torch.rand(buf_size // 4, dtype=torch.float32).cuda() for _ in range(2)]
            self._reg_descs = self.nixl_agent.get_reg_descs(self._tensors)
            assert self.nixl_agent.register_memory(self._reg_descs) is not None
            self.local_xfer_descs: nixlXferDList = self.nixl_agent.get_xfer_descs(self._tensors)
            logger.info(f"{self.name} tensors in __init__: {self._tensors}")

        # For simplicity, the test is 1:1, so just set string once receive metadata
        self._added_remote_name = None
        
    def cleanup(self):
        if self._added_remote_name:
            self.nixl_agent.remove_remote_agent(self._added_remote_name)
        self.nixl_agent.deregister_memory(self._reg_descs)
        if TEST_DRAM:
            nixl_utils.free_passthru(self._addr)

    def _get_metadata(self) -> Any:
        return self.nixl_agent.get_agent_metadata()

    def _add_remote_agent(self, agent_metadata: bytes) -> str:
        remote_name = self.nixl_agent.add_remote_agent(agent_metadata)
        logger.info(f"Loaded {remote_name} from metadata: {hash(agent_metadata)}")
        return remote_name

    def _get_serialized_descs(self, descs) -> Any:
        return self.nixl_agent.get_serialized_descs(descs)
    
    def _deserialize_descs(self, serdes) -> Any:
        src_descs_recvd = self.nixl_agent.deserialize_descs(serdes)
        logger.info("Deserialized descs hash: %s", hash(src_descs_recvd))
        return src_descs_recvd


class Target(NixlWrapper):
    def __init__(self, **kwargs):
        super().__init__(name="target", **kwargs)

    def prepare_for_read(self) -> AggregatedAgentMetadata:
        return AggregatedAgentMetadata(
            engine_id=self.name,
            agent_metadata=self._get_metadata(),
            xfer_descs=self.local_xfer_descs
        )

    def blocking_wait_for_read(self, reader_name: str, tag: bytes) -> None:
        while not self.nixl_agent.check_remote_xfer_done(remote_agent_name= reader_name,  lookup_tag=tag):
            time.sleep(0.01)

class Initiator(NixlWrapper):
    def __init__(self, **kwargs):
        super().__init__(name="initiator", **kwargs)

    def cleanup(self):
        if not TEST_DRAM:
            logger.info(f"{self.name} tensors: {self._tensors}")
        super().cleanup()

    def blocking_read_remote(self, aggregated_metadata: AggregatedAgentMetadata) -> Any:
        remote_name = self._add_remote_agent(aggregated_metadata.agent_metadata)
        self._added_remote_name = remote_name
        tag = b"UUID1"
        # vllm uses prep_xfer_* instead.
        read_handle = self.nixl_agent.initialize_xfer(
            operation="READ",
            local_descs=self.local_xfer_descs,
            remote_descs=aggregated_metadata.xfer_descs,
            remote_agent=remote_name,
            notif_msg=tag
        )
        if not read_handle:
            raise Exception(f"{self.name} failed to initialize transfer")

        # Above part can be reused, that is: read_handle could be reused once
        # target data change.

        state = self.nixl_agent.transfer(read_handle)
        assert state != "ERR", f"{state=}"
        # vllm uses check_xfer_state(handle)
        while True:
            state = self.nixl_agent.check_xfer_state(read_handle)
            if state == "ERR":
                raise Exception(f"{self.name} transfer got to Error state.")
            elif state == "DONE":
                logger.info(f"{self.name} transfer DONE")
                break
            else:
                logger.info(f"{self.name} transfer {state}")
            time.sleep(0.01)

        self.nixl_agent.release_xfer_handle(read_handle)

def test_nixl_api_no_ray():
    target = Target()
    initiator = Initiator()
    target_meta = target.prepare_for_read()

    def initiator_thread():
        initiator.blocking_read_remote(target_meta)

    def target_thread():
        target.blocking_wait_for_read(initiator.name, b"UUID1")

    import threading
    initiator_thread = threading.Thread(target=initiator_thread)
    target_thread = threading.Thread(target=target_thread)
    initiator_thread.start()
    target_thread.start()
    initiator_thread.join()
    target_thread.join()

    logger.info("Test complete")

    target.cleanup()
    initiator.cleanup()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_nixl_api_no_ray()
