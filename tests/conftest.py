"""Tests run offline and free: LLM calls are disabled; the fallback paths are what is tested."""
import os

os.environ["BID_INTEL_LLM"] = "off"
