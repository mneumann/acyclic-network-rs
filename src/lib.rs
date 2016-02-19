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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
/// New type wrapping a node index.
pub struct NodeIndex(usize);

impl NodeIndex {
    pub fn new(idx: usize) -> NodeIndex {
        NodeIndex(idx)
    }

    pub fn index(&self) -> usize {
        self.0
    }
}

#[derive(Clone, Debug)]
pub struct Link<W: Clone + Debug> {
    pub node_idx: NodeIndex,
    pub weight: W,
}

#[derive(Clone, Debug)]
pub struct Node<NT: NodeType, N: Clone + Debug, W: Clone + Debug> {
    pub node_type: NT,
    pub node_data: N,
    pub input_links: Vec<Link<W>>,
    pub output_links: Vec<Link<W>>,
}

struct CycleDetector<'a, NT: NodeType + 'a, N: Clone + Debug + 'a, W: Clone + Debug + 'a> {
    nodes: &'a [Node<NT, N, W>],
    nodes_to_visit: Vec<usize>,
    seen_nodes: FixedBitSet,
    dirty: bool,
}

impl<'a, NT: NodeType + 'a, N: Clone + Debug + 'a, W: Clone + Debug + 'a> CycleDetector<'a,
                                                                                        NT,
                                                                                        N,
                                                                                        W>
    {
    fn new(network: &'a Network<NT, N, W>) -> CycleDetector<'a, NT, N, W> {
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
            for out_link in &nodes[visit_node].output_links {
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
pub struct Network<NT: NodeType, N: Clone + Debug, W: Clone + Debug> {
    nodes: Vec<Node<NT, N, W>>,
}

impl<NT: NodeType, N: Clone + Debug, W: Clone + Debug> Network<NT, N, W> {
    pub fn new() -> Network<NT, N, W> {
        Network { nodes: Vec::new() }
    }

    pub fn nodes(&self) -> &[Node<NT, N, W>] {
        &self.nodes
    }

    pub fn add_node(&mut self, node_type: NT, node_data: N) -> NodeIndex {
        let idx = NodeIndex(self.nodes.len());
        self.nodes.push(Node {
            node_type: node_type,
            node_data: node_data,
            input_links: Vec::new(),
            output_links: Vec::new(),
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
            for link in node.output_links.iter() {
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
    pub fn add_link(&mut self, source_node_idx: NodeIndex, target_node_idx: NodeIndex, weight: W) {
        if let Err(err) = self.valid_link(source_node_idx, target_node_idx) {
            panic!(err);
        }

        self.nodes[source_node_idx.index()].output_links.push(Link {
            node_idx: target_node_idx,
            weight: weight.clone(),
        });

        self.nodes[target_node_idx.index()].input_links.push(Link {
            node_idx: source_node_idx,
            weight: weight,
        });
    }
}


#[derive(Clone, Debug)]
pub struct NetworkMap<NKEY: Ord + Clone + Debug, NT: NodeType, N: Clone + Debug, W: Clone + Debug> {
    network: Network<NT, N, W>,
    node_map: BTreeMap<NKEY, NodeIndex>,
    node_map_rev: BTreeMap<NodeIndex, NKEY>,
}

impl<NKEY: Ord + Clone + Debug, NT: NodeType, N: Clone + Debug, W: Clone + Debug> NetworkMap<NKEY,
                                                                                             NT,
                                                                                             N,
                                                                                             W>
    {
    pub fn new() -> NetworkMap<NKEY, NT, N, W> {
        NetworkMap {
            network: Network::new(),
            node_map: BTreeMap::new(),
            node_map_rev: BTreeMap::new(),
        }
    }

    pub fn nodes(&self) -> &[Node<NT, N, W>] {
        self.network.nodes()
    }

    /// Registers/creates a new node under the external key `node_key`.
    /// Panics if a node with `node_key` already exists.
    pub fn add_node(&mut self, node_key: NKEY, node_type: NT, node_data: N) {
        // XXX: Use node_map.entry()
        if self.node_map.contains_key(&node_key) {
            panic!("Duplicate node index");
        }
        let idx = self.network.add_node(node_type, node_data);
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
    pub fn add_link(&mut self, source_node_key: NKEY, target_node_key: NKEY, weight: W) {
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
        let i1 = g.add_node(NodeT::Input, ());
        let h1 = g.add_node(NodeT::Hidden, ());
        let h2 = g.add_node(NodeT::Hidden, ());
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
        let i1 = g.add_node(NodeT::Input, ());
        let o1 = g.add_node(NodeT::Output, ());
        let o2 = g.add_node(NodeT::Output, ());

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
