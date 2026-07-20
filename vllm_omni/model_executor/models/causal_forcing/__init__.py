# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# NOTE: Do not import model/pipeline classes here. The pipeline registry
# imports the topology module directly; keeping this file import-light avoids
# pulling heavy diffusion modules as a side effect.
