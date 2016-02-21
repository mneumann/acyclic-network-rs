extern crate fixedbitset;
extern crate rand;

use fixedbitset::FixedBitSet;
use rand::Rng;
use std::collections::BTreeMap;
use std::fmt::Debug;

pub trait NodeType: Clone + Debug + Send + Sized {
    /// If the node allows incoming connections
    fn accept_incoming_links(&self) -> bool;

    /// If the node allows outgoing connections
    fn accept_outgoing_links(&self) -> bool;
}

pub trait LinkWeight: Clone + Copy + Debug + Send + Sized {
}

impl LinkWeight for f64 {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
/// New type wrapping a node index.
pub struct NodeIndex(usize);

impl NodeIndex {
    #[inline(always)]
    pub fn new(idx: usize) -> NodeIndex {
        NodeIndex(idx)
    }

    #[inline(always)]
    pub fn index(&self) -> usize {
        self.0
    }
}

#[derive(Clone, Debug)]
struct Link<L: LinkWeight> {
    node_idx: NodeIndex,
    weight: L,

    // the activeness of a link has no influence on the
    // cycle detection.
    active: bool,
}

#[derive(Clone, Debug)]
pub struct Node<N: NodeType, L: LinkWeight> {
    node_type: N,
    forward_links: Vec<Link<L>>,
}

impl<N: NodeType, L: LinkWeight> Node<N, L> {
    #[inline(always)]
    pub fn node_type(&self) -> &N {
        &self.node_type
    }

    pub fn each_active_forward_link<F>(&self, mut f: F)
        where F: FnMut(NodeIndex, L)
    {
        for link in &self.forward_links {
            if link.active {
                f(link.node_idx, link.weight);
            }
        }
    }
}

struct CycleDetector<'a, N: NodeType + 'a, L: LinkWeight + 'a> {
    nodes: &'a [Node<N, L>],
    nodes_to_visit: Vec<usize>,
    seen_nodes: FixedBitSet,
    dirty: bool,
}

impl<'a, N: NodeType + 'a, L: LinkWeight + 'a> CycleDetector<'a, N, L> {
    fn new(network: &'a Network<N, L>) -> CycleDetector<'a, N, L> {
        CycleDetector {
            nodes: &network.nodes,
            nodes_to_visit: Vec::new(),
            seen_nodes: FixedBitSet::with_capacity(network.nodes.len()),
            dirty: false,
        }
    }

    // The algorithm used in `Network.link_would_cycle` and
    // `Network.find_random_unconnected_link_no_cycle`.  This is mostly extracted to avoid
    // repetetive memory allocations in `find_random_unconnected_link_no_cycle`.
    fn link_would_cycle(&mut self, source_node_idx: NodeIndex, target_node_idx: NodeIndex) -> bool {
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
            for out_link in &nodes[visit_node].forward_links {
                let next_node = out_link.node_idx.index();
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

#[derive(Clone, Debug)]
/// A directed, acylic network.
pub struct Network<N: NodeType, L: LinkWeight> {
    nodes: Vec<Node<N, L>>,
}

impl<N: NodeType, L: LinkWeight> Network<N, L> {
    pub fn new() -> Network<N, L> {
        Network { nodes: Vec::new() }
    }

    #[inline(always)]
    pub fn nodes(&self) -> &[Node<N, L>] {
        &self.nodes
    }

    pub fn add_node(&mut self, node_type: N) -> NodeIndex {
        let idx = NodeIndex(self.nodes.len());
        self.nodes.push(Node {
            node_type: node_type,
            forward_links: Vec::new(),
        });
        return idx;
    }

    /// Returns a random link between two unconnected nodes, which would not introduce
    /// a cycle. Return None is no such exists.
    pub fn find_random_unconnected_link_no_cycle<R: Rng>(&self,
                                                         rng: &mut R)
                                                         -> Option<(NodeIndex, NodeIndex)> {

        let n = self.nodes.len();

        let idx = &|i, j| i * n + j;

        let mut adj_matrix = FixedBitSet::with_capacity(n * n);

        // Build up a binary, undirected adjacency matrix of the graph.
        // Every unset bit in the adj_matrix will be a potential link.
        for (i, node) in self.nodes.iter().enumerate() {
            for link in node.forward_links.iter() {
                let j = link.node_idx.index();
                adj_matrix.insert(idx(i, j));
                // include the link of reverse direction, because this would
                // create a cycle anyway.
                adj_matrix.insert(idx(j, i));
            }
        }

        let adj_matrix = adj_matrix; // make immutable

        // We now test all potential links of every node in the graph, if it would
        // introduce a cycle. For that, we shuffle the node indices (`node_order`).
        // in random order.
        let mut node_order: Vec<_> = (0..n).into_iter().collect();
        let mut edge_order: Vec<_> = (0..n).into_iter().collect();
        rng.shuffle(&mut node_order);

        let node_order = node_order; // make immutable

        let mut cycler = CycleDetector::new(self);
        for &i in &node_order {
            rng.shuffle(&mut edge_order);
            for &j in &edge_order {
                if i != j && !adj_matrix.contains(idx(i, j)) {
                    // The link (i, j) neither is reflexive, nor exists.
                    let ni = NodeIndex(i);
                    let nj = NodeIndex(j);
                    if self.valid_link(ni, nj).is_ok() && !cycler.link_would_cycle(ni, nj) {
                        // If the link is valid and does not create a cycle, we are done!
                        return Some((ni, nj));
                    }
                }
            }
        }

        return None;
    }

    /// Returns true if the introduction of this directed link would lead towards a cycle.
    pub fn link_would_cycle(&self, source_node_idx: NodeIndex, target_node_idx: NodeIndex) -> bool {
        if source_node_idx == target_node_idx {
            return true;
        }

        CycleDetector::new(self).link_would_cycle(source_node_idx, target_node_idx)
    }

    // Check if the link is valid. Doesn't check for cycles.
    pub fn valid_link(&self,
                      source_node_idx: NodeIndex,
                      target_node_idx: NodeIndex)
                      -> Result<(), &'static str> {
        if source_node_idx == target_node_idx {
            return Err("Loops are not allowed");
        }

        if !self.nodes[source_node_idx.index()].node_type.accept_outgoing_links() {
            return Err("Node does not allow outgoing links");
        }

        if !self.nodes[target_node_idx.index()].node_type.accept_incoming_links() {
            return Err("Node does not allow incoming links");
        }

        Ok(())
    }

    // Note: Doesn't check for cycles (except in the simple reflexive case).
    pub fn add_link(&mut self, source_node_idx: NodeIndex, target_node_idx: NodeIndex, weight: L) {
        if let Err(err) = self.valid_link(source_node_idx, target_node_idx) {
            panic!(err);
        }

        self.nodes[source_node_idx.index()].forward_links.push(Link {
            node_idx: target_node_idx,
            weight: weight,
            active: true,
        });
    }

    fn set_link_status(&mut self,
                       source_node_idx: NodeIndex,
                       target_node_idx: NodeIndex,
                       active: bool)
                       -> bool {
        match self.find_link_mut(source_node_idx, target_node_idx) {
            Some(link) => {
                link.active = active;
                true
            }
            None => false,
        }
    }

    pub fn enable_link(&mut self, source_node_idx: NodeIndex, target_node_idx: NodeIndex) -> bool {
        self.set_link_status(source_node_idx, target_node_idx, true)
    }

    pub fn disable_link(&mut self, source_node_idx: NodeIndex, target_node_idx: NodeIndex) -> bool {
        self.set_link_status(source_node_idx, target_node_idx, false)
    }

    fn find_link_mut(&mut self,
                     source_node_idx: NodeIndex,
                     target_node_idx: NodeIndex)
                     -> Option<&mut Link<L>> {
        for link in self.nodes[source_node_idx.index()].forward_links.iter_mut() {
            if link.node_idx == target_node_idx {
                return Some(link);
            }
        }
        None
    }

    fn find_link_index(&self,
                       source_node_idx: NodeIndex,
                       target_node_idx: NodeIndex)
                       -> Option<usize> {
        for (i, link) in self.nodes[source_node_idx.index()].forward_links.iter().enumerate() {
            if link.node_idx == target_node_idx {
                return Some(i);
            }
        }
        None
    }

    /// Remove the first link that matches `source_node_idx` and `target_node_idx`.
    pub fn remove_link(&mut self, source_node_idx: NodeIndex, target_node_idx: NodeIndex) -> bool {
        match self.find_link_index(source_node_idx, target_node_idx) {
            None => false,
            Some(idx) => {
                let _link = self.nodes[source_node_idx.index()].forward_links.swap_remove(idx);
                true
            }
        }
    }
}


#[derive(Clone, Debug)]
pub struct NetworkMap<NKEY: Ord + Clone + Debug, N: NodeType, L: LinkWeight> {
    network: Network<N, L>,
    node_map: BTreeMap<NKEY, NodeIndex>,
    node_map_rev: BTreeMap<NodeIndex, NKEY>,
}

impl<NKEY: Ord + Clone + Debug, N: NodeType, L: LinkWeight> NetworkMap<NKEY, N, L> {
    pub fn new() -> NetworkMap<NKEY, N, L> {
        NetworkMap {
            network: Network::new(),
            node_map: BTreeMap::new(),
            node_map_rev: BTreeMap::new(),
        }
    }

    pub fn network(&self) -> &Network<N, L> {
        &self.network
    }

    pub fn nodes(&self) -> &[Node<N, L>] {
        self.network.nodes()
    }

    /// Registers/creates a new node under the external key `node_key`.
    ///
    /// # Panics
    ///
    /// If a node with `node_key` already exists.
    pub fn add_node(&mut self, node_key: NKEY, node_type: N) {
        // XXX: Use node_map.entry()
        if self.node_map.contains_key(&node_key) {
            panic!("Duplicate node index");
        }
        let idx = self.network.add_node(node_type);
        self.node_map.insert(node_key.clone(), idx);
        self.node_map_rev.insert(idx, node_key);
    }

    /// Returns a random link between two unconnected nodes, which would not introduce
    /// a cycle. Return None is no such exists.
    pub fn find_random_unconnected_link_no_cycle<R: Rng>(&self,
                                                         rng: &mut R)
                                                         -> Option<(&NKEY, &NKEY)> {
        match self.network.find_random_unconnected_link_no_cycle(rng) {
            Some((a, b)) => Some((&self.node_map_rev[&a], &self.node_map_rev[&b])),
            None => None,
        }
    }

    /// Returns true if the introduction of this directed link would lead towards a cycle.
    pub fn link_would_cycle(&self, source_node_key: NKEY, target_node_key: NKEY) -> bool {
        self.network.link_would_cycle(self.node_map[&source_node_key],
                                      self.node_map[&target_node_key])
    }

    // Check if the link is valid. Doesn't check for cycles.
    pub fn valid_link(&self,
                      source_node_key: NKEY,
                      target_node_key: NKEY)
                      -> Result<(), &'static str> {
        self.network.valid_link(self.node_map[&source_node_key],
                                self.node_map[&target_node_key])
    }

    // Note: Doesn't check for cycles (except in the simple reflexive case).
    pub fn add_link(&mut self, source_node_key: NKEY, target_node_key: NKEY, weight: L) {
        self.network.add_link(self.node_map[&source_node_key],
                              self.node_map[&target_node_key],
                              weight)
    }
}

#[cfg(test)]
mod tests {
    use rand;
    use super::{NodeType, Network};

    #[derive(Clone, Debug)]
    enum NodeT {
        Input,
        Hidden,
        Output,
    }

    impl NodeType for NodeT {
        fn accept_incoming_links(&self) -> bool {
            match *self {
                NodeT::Input => false,
                _ => true,
            }
        }
        fn accept_outgoing_links(&self) -> bool {
            match *self {
                NodeT::Output => false,
                _ => true,
            }
        }
    }

    #[test]
    fn test_cycle() {
        let mut g = Network::new();
        let i1 = g.add_node(NodeT::Input);
        let h1 = g.add_node(NodeT::Hidden);
        let h2 = g.add_node(NodeT::Hidden);
        assert_eq!(true, g.valid_link(i1, i1).is_err());
        assert_eq!(true, g.valid_link(h1, h1).is_err());

        assert_eq!(true, g.valid_link(h1, i1).is_err());
        assert_eq!(Ok(()), g.valid_link(i1, h1));
        assert_eq!(Ok(()), g.valid_link(i1, h2));
        assert_eq!(Ok(()), g.valid_link(h1, h2));

        g.add_link(i1, h1, 0.0);
        assert_eq!(true, g.link_would_cycle(h1, i1));
        assert_eq!(false, g.link_would_cycle(i1, h1));
        assert_eq!(false, g.link_would_cycle(i1, h2));
        assert_eq!(true, g.link_would_cycle(i1, i1));
        assert_eq!(false, g.link_would_cycle(h1, h2));
        assert_eq!(false, g.link_would_cycle(h2, h1));
        assert_eq!(false, g.link_would_cycle(h2, i1));

        g.add_link(h1, h2, 0.0);
        assert_eq!(true, g.link_would_cycle(h2, i1));
        assert_eq!(true, g.link_would_cycle(h1, i1));
        assert_eq!(true, g.link_would_cycle(h2, h1));
        assert_eq!(false, g.link_would_cycle(i1, h2));
    }

    #[test]
    fn test_find_random_unconnected_link_no_cycle() {
        let mut g = Network::new();
        let i1 = g.add_node(NodeT::Input);
        let o1 = g.add_node(NodeT::Output);
        let o2 = g.add_node(NodeT::Output);

        let mut rng = rand::thread_rng();

        let link = g.find_random_unconnected_link_no_cycle(&mut rng);
        assert_eq!(true, link.is_some());
        let l = link.unwrap();
        assert!((i1, o1) == l || (i1, o2) == l);

        g.add_link(i1, o2, 0.0);
        let link = g.find_random_unconnected_link_no_cycle(&mut rng);
        assert_eq!(true, link.is_some());
        assert_eq!((i1, o1), link.unwrap());

        g.add_link(i1, o1, 0.0);
        let link = g.find_random_unconnected_link_no_cycle(&mut rng);
        assert_eq!(false, link.is_some());
    }
}
