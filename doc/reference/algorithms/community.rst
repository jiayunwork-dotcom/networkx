***********
Communities
***********

.. automodule:: networkx.algorithms.community
.. currentmodule:: networkx


Bipartitions
------------
.. automodule:: networkx.algorithms.community.kernighan_lin
.. autosummary::
   :toctree: generated/

   kernighan_lin_bisection

Divisive Communities
--------------------
.. automodule:: networkx.algorithms.community.divisive
.. autosummary::
   :toctree: generated/

   edge_betweenness_partition
   edge_current_flow_betweenness_partition

K-Clique
--------
.. automodule:: networkx.algorithms.community.kclique
.. autosummary::
   :toctree: generated/

   k_clique_communities

Modularity-based communities
----------------------------
.. automodule:: networkx.algorithms.community.modularity_max
.. autosummary::
   :toctree: generated/

   greedy_modularity_communities
   naive_greedy_modularity_communities

Tree partitioning
-----------------
.. automodule:: networkx.algorithms.community.lukes
.. autosummary::
   :toctree: generated/

   lukes_partitioning

Label propagation
-----------------
.. automodule:: networkx.algorithms.community.label_propagation
.. autosummary::
   :toctree: generated/

   asyn_lpa_communities
   label_propagation_communities
   fast_label_propagation_communities

Louvain Community Detection
---------------------------
.. automodule:: networkx.algorithms.community.louvain
.. autosummary::
    :toctree: generated/

    louvain_communities
    louvain_partitions

Fluid Communities
-----------------
.. automodule:: networkx.algorithms.community.asyn_fluid
.. autosummary::
   :toctree: generated/

   asyn_fluidc

Measuring partitions
--------------------
.. automodule:: networkx.algorithms.community.quality
.. autosummary::
   :toctree: generated/

   modularity
   partition_quality

Signed networks
---------------
Networks whose edges carry both positive (support) and negative
(rejection) weights can be analyzed directly, without summing the two
kinds of evidence away.  ``modularity``, ``louvain_partitions``,
``louvain_communities`` and ``greedy_modularity_communities`` all accept
a keyword-only ``signed=True`` argument (undirected graphs only).  Each
edge weight is then split into a positive layer and a negative layer and
the partition is scored by the two-layer signed modularity

.. math::

   Q_{signed} = \frac{m^+ Q^+ - m^- Q^-}{m^+ + m^-}

where :math:`m^+`, :math:`m^-` are the total edge masses of the two
layers and :math:`Q^+`, :math:`Q^-` are the usual modularities of the
partition computed on each layer with the same ``resolution``.  The
score rewards communities that concentrate supportive edges while
keeping rejecting edges between communities; graphs with only positive
weights degenerate to the usual modularity.  Both optimizers consider
pairs of nodes and communities that have no edges between them, because
shared rejection is itself evidence for grouping.  See the ``signed``
parameter in :func:`community.modularity`,
:func:`community.louvain_communities`,
:func:`community.louvain_partitions` and
:func:`community.greedy_modularity_communities`.

Partitions via centrality measures
----------------------------------
.. automodule:: networkx.algorithms.community.centrality
.. autosummary::
   :toctree: generated/

   girvan_newman

Validating partitions
---------------------
.. automodule:: networkx.algorithms.community.community_utils
.. autosummary::
   :toctree: generated/

   is_partition
