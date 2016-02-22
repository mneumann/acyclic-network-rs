extern crate fixedbitset;
extern crate rand;

use fixedbitset::FixedBitSet;
use rand::Rng;
use std::fmt::Debug;
use std::collections::BTreeMap;

pub trait NodeType: Clone + Debug + Send + Sized {
    /// If the node allows incoming connections
    fn accept_incoming_links(&self) -> bool;

    /// If the node allows outgoing connections
    fn accept_outgoing_links(&self) -> bool;
}

/// Every node has an external node id. In contrast
/// to the `NodeIndex`, the ExternalNodeId can't become unstable
/// during node removal. The `ExternalNodeId` is also
/// used to sort the links.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ExternalNodeId(pub usize);

/// New type wrapping a node index. The node index is
/// used as an internal index into the node array.
/// It can become unstable in case of removal of nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct NodeIndex(usize);

/// New type wrapping a link index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct LinkIndex(usize);

impl NodeIndex {
    #[inline(always)]
    pub fn index(&self) -> usize {
        self.0
    }
}

impl LinkIndex {
    #[inline(always)]
    pub fn index(&self) -> usize {
        self.0
    }
}

// Wraps a Link for use in a single linked list
#[derive(Clone, Debug)]
enum LinkItem<L: Copy + Debug + Send + Sized> {
    Free {
        next: Option<LinkIndex>,
    },
    Used {
        next: Option<LinkIndex>,
        link: Link<L>,
    },
}

impl<L: Copy + Debug + Send + Sized> LinkItem<L> {
    fn set_next_used(&mut self, next_used: Option<LinkIndex>) {
        match *self {
            LinkItem::Used { ref mut next, .. } => {
                *next = next_used;
            }
            _ => panic!(),
        }
    }

    fn next_used(&self) -> Option<LinkIndex> {
        match *self {
            LinkItem::Used { next, .. } => next,
            _ => panic!(),
        }
    }

    fn ref_used(&self) -> &Link<L> {
        match *self {
            LinkItem::Used { ref link, .. } => link,
            _ => panic!(),
        }
    }

    fn next_free(&self) -> Option<LinkIndex> {
        match *self {
            LinkItem::Free { next } => next,
            _ => panic!(),
        }
    }

    fn is_used(&self) -> bool {
        match *self {
            LinkItem::Used { .. } => true,
            _ => false,
        }
    }
}

struct LinkIter {
    next_link_idx: Option<LinkIndex>,
    prev_link_idx: Option<LinkIndex>,
}

impl LinkIter {
    fn from(link_idx_opt: Option<LinkIndex>) -> LinkIter {
        LinkIter {
            next_link_idx: link_idx_opt,
            prev_link_idx: None,
        }
    }

    fn get_prev(&self) -> Option<LinkIndex> {
        self.prev_link_idx
    }

    fn next<L: Copy + Debug + Send + Sized>(&mut self,
                                            link_array: &[LinkItem<L>])
                                            -> Option<LinkIndex> {
        match self.next_link_idx {
            Some(idx) => {
                self.prev_link_idx = Some(idx);
                self.next_link_idx = link_array[idx.index()].next_used();
                return Some(idx);
            }
            None => {
                // Do not update the prev_link_idx
                return None;
            }
        }
    }
}

#[derive(Clone, Debug)]
struct Link<L: Copy + Debug + Send + Sized> {
    node_idx: NodeIndex,
    weight: L,

    // the activeness of a link has no influence on the
    // cycle detection.
    active: bool,
}

#[derive(Clone, Debug)]
pub struct Node<N: NodeType> {
    node_type: N,
    external_node_id: ExternalNodeId,
    first_link: Option<LinkIndex>,
    // in and out degree counts disabled links!
    in_degree: u32,
    out_degree: u32,
}

impl<N: NodeType> Node<N> {
    pub fn node_type(&self) -> &N {
        &self.node_type
    }

    pub fn external_node_id(&self) -> ExternalNodeId {
        self.external_node_id
    }

    pub fn in_degree(&self) -> u32 {
        self.in_degree
    }

    pub fn out_degree(&self) -> u32 {
        self.out_degree
    }

    pub fn in_out_degree(&self) -> (u32, u32) {
        (self.in_degree, self.out_degree)
    }
}

/// A directed, acylic network.
#[derive(Clone, Debug)]
pub struct Network<N: NodeType, L: Copy + Debug + Send + Sized> {
    nodes: Vec<Node<N>>,
    links: Vec<LinkItem<L>>, // XXX: Rename to link_items
    free_links: Option<LinkIndex>,
    node_count: usize,
    link_count: usize,
    external_node_id_map: BTreeMap<ExternalNodeId, NodeIndex>,
}

impl<N: NodeType, L: Copy + Debug + Send + Sized> Network<N, L> {
    pub fn new() -> Network<N, L> {
        Network {
            nodes: Vec::new(),
            links: Vec::new(),
            free_links: None,
            node_count: 0,
            link_count: 0,
            external_node_id_map: BTreeMap::new(),
        }
    }

    #[inline]
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    #[inline]
    pub fn link_count(&self) -> usize {
        self.link_count
    }

    #[inline(always)]
    pub fn node(&self, node_idx: NodeIndex) -> &Node<N> {
        &self.nodes[node_idx.index()]
    }

    #[inline(always)]
    fn node_mut(&mut self, node_idx: NodeIndex) -> &mut Node<N> {
        &mut self.nodes[node_idx.index()]
    }

    #[inline(always)]
    fn link_item(&self, link_idx: LinkIndex) -> &LinkItem<L> {
        &self.links[link_idx.index()]
    }

    #[inline(always)]
    fn link(&self, link_idx: LinkIndex) -> &Link<L> {
        match self.links[link_idx.index()] {
            LinkItem::Used { ref link, .. } => link,
            LinkItem::Free { .. } => {
                panic!();
            }
        }
    }

    #[inline(always)]
    fn link_mut(&mut self, link_idx: LinkIndex) -> &mut Link<L> {
        match self.links[link_idx.index()] {
            LinkItem::Used { ref mut link, ..} => link,
            LinkItem::Free { .. } => {
                panic!();
            }
        }
    }

    #[inline(always)]
    pub fn nodes(&self) -> &[Node<N>] {
        &self.nodes
    }

    #[inline]
    pub fn each_node_with_index<F>(&self, mut f: F)
        where F: FnMut(&Node<N>, NodeIndex)
    {
        for (i, node) in self.nodes.iter().enumerate() {
            f(node, NodeIndex(i));
        }
    }

    fn link_iter_for_node(&self, node_idx: NodeIndex) -> LinkIter {
        LinkIter::from(self.node(node_idx).first_link)
    }

    #[inline]
    pub fn each_active_forward_link_of_node<F>(&self, node_idx: NodeIndex, mut f: F)
        where F: FnMut(NodeIndex, L)
    {
        let mut link_iter = self.link_iter_for_node(node_idx);
        while let Some(link_idx) = link_iter.next(&self.links) {
            let link = self.link(link_idx);
            if link.active {
                f(link.node_idx, link.weight);
            }
        }
    }

    /// Adds a new node to the network with type `node_type` and id `external_node_id`.
    ///
    /// # Panics
    ///
    /// If a node with `external_node_id` already exists in the network.
    ///
    pub fn add_node(&mut self, node_type: N, external_node_id: ExternalNodeId) -> NodeIndex {
        let node_idx = NodeIndex(self.nodes.len());
        let old = self.external_node_id_map.insert(external_node_id, node_idx);
        assert!(old.is_none());
        self.nodes.push(Node {
            node_type: node_type,
            external_node_id: external_node_id,
            first_link: None,
            in_degree: 0,
            out_degree: 0,
        });
        self.node_count += 1;
        return node_idx;
    }

    pub fn delete_node(&mut self, _external_node_id: ExternalNodeId) {
        self.node_count -= 1;
        unimplemented!();
        // XXX
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
            let mut link_iter = LinkIter::from(node.first_link);
            while let Some(link_idx) = link_iter.next(&self.links) {
                let link = self.link(link_idx);
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
        // XXX: Remove deleted nodes
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

    pub fn lookup_node_index(&self, external_node_id: ExternalNodeId) -> Option<NodeIndex> {
        self.external_node_id_map.get(&external_node_id).map(|&i| i)
    }

    // Note: Doesn't check for cycles (except in the simple reflexive case).
    // Note that we use NodeType#in_order() to keep the list of links in order.
    // XXX: Need test cases.
    pub fn add_link(&mut self,
                    source_node_idx: NodeIndex,
                    target_node_idx: NodeIndex,
                    weight: L)
                    -> LinkIndex {
        if let Err(err) = self.valid_link(source_node_idx, target_node_idx) {
            panic!(err);
        }

        self.link_count += 1;
        self.node_mut(source_node_idx).out_degree += 1;
        self.node_mut(target_node_idx).in_degree += 1;

        let link = Link {
            node_idx: target_node_idx,
            weight: weight,
            active: true,
        };

        match self.find_link_index(source_node_idx, target_node_idx) {
            (Some(_), _) => {
                panic!("duplicate link");
            }
            (None, Some(prev_link_idx)) => {
                // we should insert the node after prev_link_idx.
                let next_link = self.link_item(prev_link_idx).next_used();
                let new_link_idx = self.allocate_link(link, next_link);
                self.links[prev_link_idx.index()].set_next_used(Some(new_link_idx));
                return new_link_idx;
            }
            (None, None) => {
                // prepend.
                let next_link = self.node(source_node_idx).first_link;
                let new_link_idx = self.allocate_link(link, next_link);
                self.node_mut(source_node_idx).first_link = Some(new_link_idx);
                return new_link_idx;
            }
        }
    }

    fn allocate_link(&mut self, link: Link<L>, next_link: Option<LinkIndex>) -> LinkIndex {
        if let Some(free_idx) = self.free_links {
            let next_free_idx = self.link_item(free_idx).next_free();
            self.free_links = next_free_idx;

            if let LinkItem::Used { .. } = self.links[free_idx.index()] {
                panic!("Trying to reuse an link item which is in use!");
            }

            self.links[free_idx.index()] = LinkItem::Used {
                link: link,
                next: next_link,
            };

            return free_idx;
        } else {
            let new_link_idx = LinkIndex(self.links.len());
            self.links.push(LinkItem::Used {
                link: link,
                next: next_link,
            });
            return new_link_idx;
        }
    }

    fn set_link_status(&mut self,
                       source_node_idx: NodeIndex,
                       target_node_idx: NodeIndex,
                       active: bool)
                       -> bool {
        match self.find_link_index(source_node_idx, target_node_idx) {
            (Some(link_idx), _) => {
                self.link_mut(link_idx).active = active;
                true
            }
            _ => false,
        }
    }

    pub fn enable_link(&mut self, source_node_idx: NodeIndex, target_node_idx: NodeIndex) -> bool {
        self.set_link_status(source_node_idx, target_node_idx, true)
    }

    pub fn disable_link(&mut self, source_node_idx: NodeIndex, target_node_idx: NodeIndex) -> bool {
        self.set_link_status(source_node_idx, target_node_idx, false)
    }

    // Returns (Some(index), _) if the given link was found.
    // Returns (None, _) if the link was not found.
    // (_, Some(prev_index)) is the index of the previous link in the list.
    // (_, None) when no previous element exists (list is empty or contains one element).
    fn find_link_index(&self,
                       source_node_idx: NodeIndex,
                       target_node_idx: NodeIndex)
                       -> (Option<LinkIndex>, Option<LinkIndex>) {
        // we use the target_node's external node id as sort order.
        let target_node_ext_id = self.node(target_node_idx).external_node_id();

        let mut link_iter = self.link_iter_for_node(source_node_idx);
        while let Some(link_idx) = link_iter.next(&self.links) {
            let link = self.link(link_idx);

            let link_ext_id = self.node(link.node_idx).external_node_id();

            // We found the node we are looking for.
            if link.node_idx == target_node_idx {
                debug_assert!(link_ext_id == target_node_ext_id);
                return (Some(link_idx), link_iter.get_prev());
            }
            debug_assert!(link_ext_id != target_node_ext_id);

            // We keep the links sorted according to the target node's `external_node_id`.
            // If we are out of order, the link does not exist and a new link should be
            // inserted after the prev_link.
            if link_ext_id > target_node_ext_id {
                break;
            }
        }
        return (None, link_iter.get_prev());
    }

    /// Remove the first link that matches `source_node_idx` and `target_node_idx`.
    pub fn remove_link(&mut self, source_node_idx: NodeIndex, target_node_idx: NodeIndex) -> bool {
        let found_idx = match self.find_link_index(source_node_idx, target_node_idx) {
            (Some(found_idx), Some(prev_idx)) => {
                let next_used = self.link_item(found_idx).next_used();
                self.links[prev_idx.index()].set_next_used(next_used);
                found_idx
            }
            (Some(found_idx), None) => {
                // `found_idx` is the first item in the list.
                assert!(self.node(source_node_idx).first_link == Some(found_idx));
                assert!(self.link_item(found_idx).next_used().is_none());
                self.node_mut(source_node_idx).first_link = None;
                found_idx
            }
            (None, _) => {
                // link was not found
                return false;
            }
        };

        // push found_idx on free list
        assert!(self.links[found_idx.index()].is_used());
        self.links[found_idx.index()] = LinkItem::Free { next: self.free_links };
        self.free_links = Some(found_idx);

        assert!(self.node(source_node_idx).out_degree > 0);
        assert!(self.node(target_node_idx).in_degree > 0);
        self.node_mut(source_node_idx).out_degree -= 1;
        self.node_mut(target_node_idx).in_degree -= 1;
        self.link_count -= 1;
        return true;
    }
}

struct CycleDetector<'a, N: NodeType + 'a, L: Copy + Debug + Send + Sized + 'a> {
    nodes: &'a [Node<N>],
    links: &'a [LinkItem<L>],
    nodes_to_visit: Vec<usize>,
    seen_nodes: FixedBitSet,
    dirty: bool,
}

impl<'a, N: NodeType + 'a, L: Copy + Debug + Send + Sized + 'a> CycleDetector<'a, N, L> {
    fn new(network: &'a Network<N, L>) -> CycleDetector<'a, N, L> {
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
            let mut link_iter = LinkIter::from(nodes[visit_node].first_link);

            while let Some(link_idx) = link_iter.next(self.links) {
                let out_link = self.links[link_idx.index()].ref_used();
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

#[cfg(test)]
mod tests {
    use rand;
    use super::{NodeType, Network, ExternalNodeId};

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
        let i1 = g.add_node(NodeT::Input, ExternalNodeId(1));
        let h1 = g.add_node(NodeT::Hidden, ExternalNodeId(2));
        let h2 = g.add_node(NodeT::Hidden, ExternalNodeId(3));
        assert_eq!(true, g.valid_link(i1, i1).is_err());
        assert_eq!(true, g.valid_link(h1, h1).is_err());

        assert_eq!(true, g.valid_link(h1, i1).is_err());
        assert_eq!(Ok(()), g.valid_link(i1, h1));
        assert_eq!(Ok(()), g.valid_link(i1, h2));
        assert_eq!(Ok(()), g.valid_link(h1, h2));

        assert_eq!((0, 0), g.node(i1).in_out_degree());
        assert_eq!((0, 0), g.node(h1).in_out_degree());
        assert_eq!((0, 0), g.node(h2).in_out_degree());

        g.add_link(i1, h1, 0.0);
        assert_eq!((0, 1), g.node(i1).in_out_degree());
        assert_eq!((1, 0), g.node(h1).in_out_degree());
        assert_eq!((0, 0), g.node(h2).in_out_degree());

        assert_eq!(true, g.link_would_cycle(h1, i1));
        assert_eq!(false, g.link_would_cycle(i1, h1));
        assert_eq!(false, g.link_would_cycle(i1, h2));
        assert_eq!(true, g.link_would_cycle(i1, i1));
        assert_eq!(false, g.link_would_cycle(h1, h2));
        assert_eq!(false, g.link_would_cycle(h2, h1));
        assert_eq!(false, g.link_would_cycle(h2, i1));

        g.add_link(h1, h2, 0.0);
        assert_eq!((0, 1), g.node(i1).in_out_degree());
        assert_eq!((1, 1), g.node(h1).in_out_degree());
        assert_eq!((1, 0), g.node(h2).in_out_degree());

        assert_eq!(true, g.link_would_cycle(h2, i1));
        assert_eq!(true, g.link_would_cycle(h1, i1));
        assert_eq!(true, g.link_would_cycle(h2, h1));
        assert_eq!(false, g.link_would_cycle(i1, h2));
    }

    #[test]
    fn test_lookup_ext_id() {
        let mut g: Network<_, f64> = Network::new();
        let i1 = g.add_node(NodeT::Input, ExternalNodeId(1));
        let h1 = g.add_node(NodeT::Hidden, ExternalNodeId(2));
        let h2 = g.add_node(NodeT::Hidden, ExternalNodeId(3));
        assert_eq!(Some(i1), g.lookup_node_index(ExternalNodeId(1)));
        assert_eq!(Some(h1), g.lookup_node_index(ExternalNodeId(2)));
        assert_eq!(Some(h2), g.lookup_node_index(ExternalNodeId(3)));
        assert_eq!(None, g.lookup_node_index(ExternalNodeId(4)));
    }

    #[test]
    fn test_find_random_unconnected_link_no_cycle() {
        let mut g = Network::new();
        let i1 = g.add_node(NodeT::Input, ExternalNodeId(1));
        let o1 = g.add_node(NodeT::Output, ExternalNodeId(2));
        let o2 = g.add_node(NodeT::Output, ExternalNodeId(3));

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
