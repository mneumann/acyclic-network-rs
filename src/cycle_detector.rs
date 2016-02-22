use fixedbitset::FixedBitSet;
use std::fmt::Debug;
use super::{NodeType, Node, LinkItem, Network, NodeIndex, LinkIter};

pub struct CycleDetector<'a,
                     N: NodeType + 'a,
                     L: Copy + Debug + Send + Sized + 'a,
                     EXTID: Copy + Debug + Send + Sized + Ord + 'a>
{
    nodes: &'a [Node<N, EXTID>],
    links: &'a [LinkItem<L, EXTID>],
    nodes_to_visit: Vec<usize>,
    seen_nodes: FixedBitSet,
    dirty: bool,
}

impl<'a, N: NodeType + 'a, L: Copy + Debug + Send + Sized + 'a, EXTID: Copy + Debug + Send + Sized + Ord + 'a> CycleDetector<'a, N, L, EXTID> {
    pub fn new(network: &'a Network<N, L, EXTID>) -> CycleDetector<'a, N, L, EXTID> {
        CycleDetector {
            nodes: &network.nodes,
            links: &network.links,
            nodes_to_visit: Vec::new(),
            seen_nodes: FixedBitSet::with_capacity(network.nodes.len()),
            dirty: false,
        }
    }

    // The algorithm used in `Network.link_would_cycle` and
    // `Network.find_random_unconnected_link_no_cycle`.  This is mostly extracted to avoid
    // repetetive memory allocations in `find_random_unconnected_link_no_cycle`.
    pub fn link_would_cycle(&mut self, source_node_idx: NodeIndex, target_node_idx: NodeIndex) -> bool {
        let path_from = target_node_idx.index();
        let path_to = source_node_idx.index();

        assert!(path_from != path_to);

        let nodes = self.nodes;
        let mut nodes_to_visit = &mut self.nodes_to_visit;
        let mut seen_nodes = &mut self.seen_nodes;

        if self.dirty {
            nodes_to_visit.clear();
            seen_nodes.clear();
        }
        self.dirty = true;

        // We start at the from the target_node and iterate all paths from there. If we hit the source node,
        // the addition of this link would lead towards a cycle. Otherwise not.
        nodes_to_visit.push(path_from);
        seen_nodes.insert(path_from);

        while let Some(visit_node) = nodes_to_visit.pop() {
            for (_, out_link) in LinkIter::new(nodes[visit_node].first_link, self.links) {
                let next_node = out_link.target_node_idx.index();
                if !seen_nodes.contains(next_node) {
                    if next_node == path_to {
                        // We found a path to `path_to`. We have found a cycle.
                        return true;
                    }

                    seen_nodes.insert(next_node);
                    nodes_to_visit.push(next_node)
                }
            }
        }

        // We haven't found a cycle.
        return false;
    }
}
