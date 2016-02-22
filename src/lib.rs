extern crate fixedbitset;
extern crate rand;

mod cycle_detector;

use cycle_detector::CycleDetector;
use fixedbitset::FixedBitSet;
use rand::Rng;
use std::fmt::Debug;

pub trait NodeType: Clone + Debug + Send + Sized {
    /// If the node allows incoming connections
    fn accept_incoming_links(&self) -> bool;

    /// If the node allows outgoing connections
    fn accept_outgoing_links(&self) -> bool;
}

/// Every node or link contains an external id. In contrast
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ExternalId(pub usize);

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

// Wraps a Link for use in a double linked list
#[derive(Clone, Debug)]
struct LinkItem<L, EXTID>
    where L: Copy + Debug + Send + Sized,
          EXTID: Copy + Debug + Send + Sized + Ord
{
    prev: Option<LinkIndex>,
    next: Option<LinkIndex>,
    link: Link<L, EXTID>,
}

pub struct LinkIter<'a, L, EXTID>
    where L: Copy + Debug + Send + Sized + 'a,
          EXTID: Copy + Debug + Send + Sized + Ord + 'a
{
    next_link_idx: Option<LinkIndex>,
    prev_link_idx: Option<LinkIndex>,
    link_array: &'a [LinkItem<L, EXTID>],
}

impl<'a, L, EXTID> LinkIter<'a, L, EXTID>
    where L: Copy + Debug + Send + Sized + 'a,
          EXTID: Copy + Debug + Send + Sized + Ord + 'a
{
    fn new(link_idx_opt: Option<LinkIndex>, link_array: &'a [LinkItem<L, EXTID>]) -> Self {
        LinkIter {
            next_link_idx: link_idx_opt,
            prev_link_idx: None,
            link_array: link_array,
        }
    }

    fn get_prev(&self) -> Option<LinkIndex> {
        self.prev_link_idx
    }
}

impl<'a, L, EXTID> Iterator for LinkIter<'a, L, EXTID>
    where L: Copy + Debug + Send + Sized + 'a,
          EXTID: Copy + Debug + Send + Sized + Ord + 'a
{
    type Item = (LinkIndex, &'a Link<L, EXTID>);

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_link_idx {
            Some(idx) => {
                self.prev_link_idx = Some(idx);
                let item = &self.link_array[idx.index()];
                self.next_link_idx = item.next;
                return Some((idx, &item.link));
            }
            None => {
                // Do not update the prev_link_idx
                return None;
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct Link<L, EXTID>
    where L: Copy + Debug + Send + Sized,
          EXTID: Copy + Debug + Send + Sized + Ord
{
    source_node_idx: NodeIndex,
    target_node_idx: NodeIndex,
    weight: L,
    external_link_id: EXTID,

    // the activeness of a link has no influence on the
    // cycle detection.
    active: bool,
}

impl<L, EXTID> Link<L, EXTID>
    where L: Copy + Debug + Send + Sized,
          EXTID: Copy + Debug + Send + Sized + Ord
{
    pub fn external_link_id(&self) -> EXTID {
        self.external_link_id
    }

    pub fn is_active(&self) -> bool {
        self.active
    }
}

#[derive(Clone, Debug)]
struct List {
    head: Option<LinkIndex>,
}

impl List {
    fn iter<'a, L, EXTID>(&self, link_array: &'a [LinkItem<L, EXTID>]) -> LinkIter<'a, L, EXTID>
        where L: Copy + Debug + Send + Sized + 'a,
              EXTID: Copy + Debug + Send + Sized + Ord + 'a
    {
        LinkIter::new(self.head, link_array)
    }
}

#[derive(Clone, Debug)]
pub struct Node<N: NodeType, EXTID: Copy + Debug + Send + Sized + Ord = ExternalId> {
    node_type: N,
    external_node_id: EXTID,
    links: List,
    // in and out degree counts disabled links!
    in_degree: u32,
    out_degree: u32,
}

impl<N: NodeType, EXTID: Copy + Debug + Send + Sized + Ord = ExternalId> Node<N, EXTID> {
    pub fn node_type(&self) -> &N {
        &self.node_type
    }

    pub fn external_node_id(&self) -> EXTID {
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
pub struct Network<N: NodeType,
                   L: Copy + Debug + Send + Sized,
                   EXTID: Copy + Debug + Send + Sized + Ord = ExternalId>
{
    nodes: Vec<Node<N, EXTID>>,
    links: Vec<LinkItem<L, EXTID>>, // XXX: Rename to link_items
    node_count: usize,
    link_count: usize,
}

impl<N: NodeType, L: Copy + Debug + Send + Sized, EXTID: Copy + Debug + Send + Sized + Ord = ExternalId> Network<N, L, EXTID> {
    pub fn new() -> Network<N, L, EXTID> {
        Network {
            nodes: Vec::new(),
            links: Vec::new(),
            node_count: 0,
            link_count: 0,
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
    pub fn node(&self, node_idx: NodeIndex) -> &Node<N, EXTID> {
        &self.nodes[node_idx.index()]
    }

    #[inline(always)]
    fn node_mut(&mut self, node_idx: NodeIndex) -> &mut Node<N, EXTID> {
        &mut self.nodes[node_idx.index()]
    }

    #[inline(always)]
    fn link_item(&self, link_idx: LinkIndex) -> &LinkItem<L, EXTID> {
        &self.links[link_idx.index()]
    }

    #[inline(always)]
    fn link(&self, link_idx: LinkIndex) -> &Link<L, EXTID> {
        &self.link_item(link_idx).link
    }

    #[inline(always)]
    fn link_mut(&mut self, link_idx: LinkIndex) -> &mut Link<L, EXTID> {
        &mut(self.links[link_idx.index()].link)
    }

    #[inline(always)]
    pub fn nodes(&self) -> &[Node<N, EXTID>] {
        &self.nodes
    }

    #[inline]
    pub fn each_node_with_index<F>(&self, mut f: F)
        where F: FnMut(&Node<N, EXTID>, NodeIndex)
    {
        for (i, node) in self.nodes.iter().enumerate() {
            f(node, NodeIndex(i));
        }
    }

    #[inline]
    pub fn link_iter_for_node<'a>(&'a self, node_idx: NodeIndex) -> LinkIter<'a, L, EXTID> {
        self.node(node_idx).links.iter(&self.links)
    }

    #[inline]
    pub fn each_active_forward_link_of_node<F>(&self, node_idx: NodeIndex, mut f: F)
        where F: FnMut(NodeIndex, L)
    {
        for (_, link) in self.link_iter_for_node(node_idx) {
            if link.active {
                f(link.target_node_idx, link.weight);
            }
        }
    }

/// Adds a new node to the network with type `node_type` and the associated
/// id `external_node_id`. The `external_node_id` is stored in the node and
/// can be retrieved later on.
///
    pub fn add_node(&mut self, node_type: N, external_node_id: EXTID) -> NodeIndex {
        let node_idx = NodeIndex(self.nodes.len());
        self.nodes.push(Node {
            node_type: node_type,
            external_node_id: external_node_id,
            links: List {head: None},
            in_degree: 0,
            out_degree: 0,
        });
        self.node_count += 1;
        return node_idx;
    }

    pub fn delete_node(&mut self, _node_idx: NodeIndex) {
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
            for (_, link) in node.links.iter(&self.links) {
                let j = link.target_node_idx.index();
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

    fn allocate_link_item(&mut self, link_item: LinkItem<L, EXTID>) -> LinkIndex {
        let new_link_idx = LinkIndex(self.links.len());
        self.links.push(link_item);
        return new_link_idx;
    }

    fn set_link_status(&mut self,
                       source_node_idx: NodeIndex,
                       target_node_idx: NodeIndex,
                       active: bool)
                       -> bool {
        match self.find_link_index_exact(source_node_idx, target_node_idx) {
            Some(link_idx) => {
                self.link_mut(link_idx).active = active;
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

    // Note: Doesn't check for cycles (except in the simple reflexive case).
// Note that we keep the list of links sorted according to it's external_link_id.
// XXX: Need test cases.
    pub fn add_link(&mut self,
                    source_node_idx: NodeIndex,
                    target_node_idx: NodeIndex,
                    weight: L,
                    external_link_id: EXTID)
                    -> LinkIndex {
        if let Err(err) = self.valid_link(source_node_idx, target_node_idx) {
            panic!(err);
        }

        self.link_count += 1;
        self.node_mut(source_node_idx).out_degree += 1;
        self.node_mut(target_node_idx).in_degree += 1;

        let link = Link {
            source_node_idx: source_node_idx,
            target_node_idx: target_node_idx,
            external_link_id: external_link_id,
            weight: weight,
            active: true,
        };

        match self.find_link_index_insert(source_node_idx, target_node_idx, external_link_id) {
            None => {
// prepend
                let next_link = self.node(source_node_idx).links.head;
                let new_link_idx = self.allocate_link_item(LinkItem{
                    link: link,
                    prev: None,
                    next: next_link,
                });
                self.node_mut(source_node_idx).links.head = Some(new_link_idx);
                return new_link_idx;
            }
            Some(idx) => {
                if self.link(idx).target_node_idx == target_node_idx {
                    assert!(self.link(idx).external_link_id == external_link_id);
                    panic!("Duplicate link");
                }
                let next_link = self.link_item(idx).next;
// insert new link after `idx`
                let new_link_idx = self.allocate_link_item(LinkItem{
                    link: link,
                    prev: Some(idx),
                    next: next_link,
                });
                self.links[idx.index()].next = Some(new_link_idx);
                return new_link_idx;
            }
        }
    }

// returns the index of the element whoose external link id is <= than
// `external_link_id`.
    fn find_link_index_insert(&self,
                       source_node_idx: NodeIndex,
                       _target_node_idx: NodeIndex,
                       external_link_id: EXTID)
                       -> Option<LinkIndex> {
// the links are sorted according to their external link id.
        let mut link_iter = self.link_iter_for_node(source_node_idx);
        for (_, link) in &mut link_iter {
            if link.external_link_id() > external_link_id {
                break;
            }
        }
        return link_iter.get_prev();
    }

    fn find_link_index_exact(&self,
                       source_node_idx: NodeIndex,
                       target_node_idx: NodeIndex)
                       -> Option<LinkIndex> {
        for (link_idx, link) in self.link_iter_for_node(source_node_idx) {
// We found the node we are looking for.
            if link.target_node_idx == target_node_idx {
                return Some(link_idx);
            }
        }
        return None;
    }

/// Remove the first link that matches `source_node_idx` and `target_node_idx`.
    pub fn remove_link(&mut self, source_node_idx: NodeIndex, target_node_idx: NodeIndex) -> bool {
        match self.find_link_index_exact(source_node_idx, target_node_idx) {
            Some(found_idx) => {
                debug_assert!(self.link(found_idx).source_node_idx == source_node_idx);
                debug_assert!(self.link(found_idx).target_node_idx == target_node_idx);
// remove item from chain
                match self.link_item(found_idx).prev {
                    None => {
// `found_idx` is the first element in the list
                        assert!(self.node(source_node_idx).links.head == Some(found_idx));
                        self.node_mut(source_node_idx).links.head = self.link_item(found_idx).next;
                    }
                    Some(prev_idx) => {
                        assert!(self.link_item(prev_idx).next == Some(found_idx));
                        let next = self.link_item(found_idx).next;
                        self.links[prev_idx.index()].next = next;
                    }
                }

                match self.link_item(found_idx).next {
                    None => {
// nothing to do
                    }
                    Some(next_idx) => {
                        assert!(self.link_item(next_idx).prev == Some(found_idx));
                        let prev = self.link_item(found_idx).prev;
                        self.links[next_idx.index()].prev = prev;
                    }
                }

                self.links[found_idx.index()].next = None;
                self.links[found_idx.index()].prev = None;

// swap the item with the last one.
                let replaced_element_idx = LinkIndex(self.links.len() - 1);

                if found_idx == replaced_element_idx {
// if we are the last element, we can just pop it
                    let old = self.links.pop().unwrap();
                    debug_assert!(old.link.source_node_idx == source_node_idx);
                    debug_assert!(old.link.target_node_idx == target_node_idx);
                } else {
                    let old = self.links.swap_remove(found_idx.index());
                    debug_assert!(old.link.source_node_idx == source_node_idx);
                    debug_assert!(old.link.target_node_idx == target_node_idx);

// We have to change the linking of the newly at position `found_idx` placed
// element.
                    let new_idx = found_idx;
// change the next pointer of the previous element
                    match self.link_item(new_idx).prev {
                        Some(idx) => {
                            assert!(self.link_item(idx).next == Some(replaced_element_idx));
                            self.links[idx.index()].next = Some(new_idx);
                        }
                        None => {
// it was the first in the chain
// we have to update the associated nodes `links.head` field.
                            assert!(self.node(self.link(new_idx).source_node_idx).links.head == Some(replaced_element_idx));
                            let src = self.link(new_idx).source_node_idx;
                            self.node_mut(src).links.head = Some(new_idx);
                        }
                    }

// change the prev pointer of the next element
                    match self.link_item(new_idx).next {
                        Some(idx) => {
                            assert!(self.link_item(idx).prev == Some(replaced_element_idx));
                            self.links[idx.index()].prev = Some(new_idx);
                        }
                        None => {
// last in the chain. nothing to do.
                        }
                    }
                }

                assert!(self.node(source_node_idx).out_degree > 0);
                assert!(self.node(target_node_idx).in_degree > 0);
                self.node_mut(source_node_idx).out_degree -= 1;
                self.node_mut(target_node_idx).in_degree -= 1;
                self.link_count -= 1;
                return true;
            }

            None => {
// link was not found
               return false;
            }
        }

    }
}

#[cfg(test)]
mod tests {
    use rand;
    use super::{NodeType, Network, ExternalId};

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
        let i1 = g.add_node(NodeT::Input, ExternalId(1));
        let h1 = g.add_node(NodeT::Hidden, ExternalId(2));
        let h2 = g.add_node(NodeT::Hidden, ExternalId(3));
        assert_eq!(true, g.valid_link(i1, i1).is_err());
        assert_eq!(true, g.valid_link(h1, h1).is_err());

        assert_eq!(true, g.valid_link(h1, i1).is_err());
        assert_eq!(Ok(()), g.valid_link(i1, h1));
        assert_eq!(Ok(()), g.valid_link(i1, h2));
        assert_eq!(Ok(()), g.valid_link(h1, h2));

        assert_eq!((0, 0), g.node(i1).in_out_degree());
        assert_eq!((0, 0), g.node(h1).in_out_degree());
        assert_eq!((0, 0), g.node(h2).in_out_degree());

        g.add_link(i1, h1, 0.0, ExternalId(1));
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

        g.add_link(h1, h2, 0.0, ExternalId(2));
        assert_eq!((0, 1), g.node(i1).in_out_degree());
        assert_eq!((1, 1), g.node(h1).in_out_degree());
        assert_eq!((1, 0), g.node(h2).in_out_degree());

        assert_eq!(true, g.link_would_cycle(h2, i1));
        assert_eq!(true, g.link_would_cycle(h1, i1));
        assert_eq!(true, g.link_would_cycle(h2, h1));
        assert_eq!(false, g.link_would_cycle(i1, h2));
    }

    #[test]
    fn test_find_random_unconnected_link_no_cycle() {
        let mut g = Network::new();
        let i1 = g.add_node(NodeT::Input, ExternalId(1));
        let o1 = g.add_node(NodeT::Output, ExternalId(2));
        let o2 = g.add_node(NodeT::Output, ExternalId(3));

        let mut rng = rand::thread_rng();

        let link = g.find_random_unconnected_link_no_cycle(&mut rng);
        assert_eq!(true, link.is_some());
        let l = link.unwrap();
        assert!((i1, o1) == l || (i1, o2) == l);

        g.add_link(i1, o2, 0.0, ExternalId(1));
        let link = g.find_random_unconnected_link_no_cycle(&mut rng);
        assert_eq!(true, link.is_some());
        assert_eq!((i1, o1), link.unwrap());

        g.add_link(i1, o1, 0.0, ExternalId(2));
        let link = g.find_random_unconnected_link_no_cycle(&mut rng);
        assert_eq!(false, link.is_some());
    }

    #[test]
    fn test_remove_link() {
        let mut g = Network::new();
        let i1 = g.add_node(NodeT::Input, ExternalId(1));
        let h1 = g.add_node(NodeT::Hidden, ExternalId(2));
        let h2 = g.add_node(NodeT::Hidden, ExternalId(3));

        g.add_link(i1, h1, 0.0, ExternalId(2));
        g.add_link(i1, h2, 0.0, ExternalId(1));

        assert_eq!(2, g.node(i1).out_degree());
        assert_eq!(1, g.node(h1).in_degree());
        assert_eq!(1, g.node(h2).in_degree());
        assert_eq!(2, g.link_count());

        assert_eq!(true, g.remove_link(i1, h1));
        assert_eq!(1, g.node(i1).out_degree());
        assert_eq!(0, g.node(h1).in_degree());
        assert_eq!(1, g.node(h2).in_degree());
        assert_eq!(1, g.link_count());

        assert_eq!(false, g.remove_link(i1, h1));
        assert_eq!(1, g.node(i1).out_degree());
        assert_eq!(0, g.node(h1).in_degree());
        assert_eq!(1, g.node(h2).in_degree());
        assert_eq!(1, g.link_count());

        assert_eq!(true, g.remove_link(i1, h2));
        assert_eq!(0, g.node(i1).out_degree());
        assert_eq!(0, g.node(h1).in_degree());
        assert_eq!(0, g.node(h2).in_degree());
        assert_eq!(0, g.link_count());

        // XXX: test for sort order
    }


}
