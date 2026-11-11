"""Tests run offline and free: LLM calls and Langfuse tracing are disabled; fallbacks are tested."""
import os

os.environ["BID_INTEL_LLM"] = "off"
os.environ["BID_INTEL_TRACING"] = "off"
