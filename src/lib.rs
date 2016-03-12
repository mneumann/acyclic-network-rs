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
    link_array: &'a [LinkItem<L, EXTID>],
}

impl<'a, L, EXTID> LinkIter<'a, L, EXTID>
    where L: Copy + Debug + Send + Sized + 'a,
          EXTID: Copy + Debug + Send + Sized + Ord + 'a
{
    fn new(link_idx_opt: Option<LinkIndex>, link_array: &'a [LinkItem<L, EXTID>]) -> Self {
        LinkIter {
            next_link_idx: link_idx_opt,
            link_array: link_array,
        }
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
                let item = &self.link_array[idx.index()];
                self.next_link_idx = item.next;
                return Some((idx, &item.link));
            }
            None => {
                return None;
            }
        }
    }
}


pub struct LinkRefIter<'a, N, L, EXTID>
    where N: NodeType + 'a,
          L: Copy + Debug + Send + Sized + 'a,
          EXTID: Copy + Debug + Send + Sized + Ord + 'a
{
    next_link_idx: Option<LinkIndex>,
    network: &'a Network<N, L, EXTID>,
}

/// A LinkRefItem includes a pointer to the network, 
/// as such, it is read only.
pub struct LinkRefItem<'a, N, L, EXTID>
    where N: NodeType + 'a,
          L: Copy + Debug + Send + Sized + 'a,
          EXTID: Copy + Debug + Send + Sized + Ord + 'a
{
    link: &'a Link<L, EXTID>,
    network: &'a Network<N, L, EXTID>,
}

impl<'a, N, L, EXTID> LinkRefItem<'a, N, L, EXTID>
    where N: NodeType + 'a,
          L: Copy + Debug + Send + Sized + 'a,
          EXTID: Copy + Debug + Send + Sized + Ord + 'a
{
    pub fn link(&self) -> &Link<L, EXTID> {
        self.link
    }

    pub fn network(&self) -> &Network<N, L, EXTID> {
        self.network
    }

    pub fn external_link_id(&self) -> EXTID {
        self.link.external_link_id()
    }

    pub fn source_node(&self) -> &Node<N, EXTID> {
        self.network.node(self.link.source_node_idx)
    }

    pub fn target_node(&self) -> &Node<N, EXTID> {
        self.network.node(self.link.target_node_idx)
    }

    pub fn external_source_node_id(&self) -> EXTID {
        self.source_node().external_node_id()
    }

    pub fn external_target_node_id(&self) -> EXTID {
        self.target_node().external_node_id()
    }
}

impl<'a, N, L, EXTID> LinkRefIter<'a, N, L, EXTID>
    where N: NodeType + 'a,
          L: Copy + Debug + Send + Sized + 'a,
          EXTID: Copy + Debug + Send + Sized + Ord + 'a
{
    fn new(link_idx_opt: Option<LinkIndex>, network: &'a Network<N, L, EXTID>) -> Self {
        LinkRefIter {
            next_link_idx: link_idx_opt,
            network: network,
        }
    }
}

impl<'a, N, L, EXTID> Iterator for LinkRefIter<'a, N, L, EXTID>
    where N: NodeType + 'a,
          L: Copy + Debug + Send + Sized + 'a,
          EXTID: Copy + Debug + Send + Sized + Ord + 'a
{
    type Item = (LinkRefItem<'a, N, L, EXTID>);

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_link_idx {
            Some(idx) => {
                let item = &self.network.links[idx.index()];
                self.next_link_idx = item.next;
                return Some(LinkRefItem {
                    link: &item.link,
                    network: &self.network,
                });
            }
            None => {
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
    #[inline(always)]
    pub fn external_link_id(&self) -> EXTID {
        self.external_link_id
    }

    #[inline(always)]
    pub fn source_node_index(&self) -> NodeIndex {
        self.source_node_idx
    }

    #[inline(always)]
    pub fn target_node_index(&self) -> NodeIndex {
        self.target_node_idx
    }

    #[inline(always)]
    pub fn is_active(&self) -> bool {
        self.active
    }

    #[inline(always)]
    pub fn weight(&self) -> L {
        self.weight
    }

    #[inline(always)]
    pub fn set_weight(&mut self, new_weight: L) {
        self.weight = new_weight;
    }
}

#[derive(Clone, Debug)]
struct List {
    head: Option<LinkIndex>,
    tail: Option<LinkIndex>,
}

impl List {
    fn empty() -> Self {
        List {
            head: None,
            tail: None,
        }
    }

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
    active_link_count: usize,
}

impl<N: NodeType, L: Copy + Debug + Send + Sized, EXTID: Copy + Debug + Send + Sized + Ord = ExternalId> Network<N, L, EXTID> {
    pub fn new() -> Network<N, L, EXTID> {
        Network {
            nodes: Vec::new(),
            links: Vec::new(),
            node_count: 0,
            link_count: 0,
            active_link_count: 0,
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
    pub fn link(&self, link_idx: LinkIndex) -> &Link<L, EXTID> {
        &self.link_item(link_idx).link
    }

    #[inline(always)]
    pub fn link_mut(&mut self, link_idx: LinkIndex) -> &mut Link<L, EXTID> {
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
    pub fn link_ref_iter_for_node<'a>(&'a self, node_idx: NodeIndex) -> LinkRefIter<'a, N, L, EXTID> {
        LinkRefIter::new(self.node(node_idx).links.head, self)
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

    #[inline]
    pub fn each_link_ref<F>(&self, mut f: F)
        where F: FnMut(LinkRefItem<N, L, EXTID>)
    {
        for link_item in &self.links[..] {
            f(LinkRefItem{link: &&link_item.link, network: self});
        }
    }

    #[inline]
    pub fn each_link_mut<F>(&mut self, mut f: F)
        where F: FnMut(&mut Link<L, EXTID>)
    {
        for link_item in &mut self.links[..] {
            f(&mut link_item.link);
        }
    }

    fn inactive_link_count(&self) -> usize {
        assert!(self.link_count >= self.active_link_count);
        self.link_count - self.active_link_count
    }

/// # Complexity
///
/// O(number of links)

    pub fn random_inactive_link_index<R: Rng>(&self, rng: &mut R) -> Option<LinkIndex> {
        let n = self.inactive_link_count();
        assert!(n <= self.link_count);
        if n > 0 {
            let mut nth_link: usize = rng.gen_range(0, n);

            for node in self.nodes.iter() {
                for (link_idx, link) in node.links.iter(&self.links) {
                    if !link.is_active() {
                        if nth_link > 0 {
                            nth_link -= 1;
                        } else {
                            return Some(link_idx);
                        }
                    }
                }
            }
        }

        return None;
    }



/// # Complexity
///
/// O(number of links)

    pub fn random_active_link_index<R: Rng>(&self, rng: &mut R) -> Option<LinkIndex> {
        let n = self.active_link_count;
        assert!(n <= self.link_count);
        if n > 0 {
            let mut nth_link: usize = rng.gen_range(0, n);

            for node in self.nodes.iter() {
                for (link_idx, link) in node.links.iter(&self.links) {
                    if link.is_active() {
                        if nth_link > 0 {
                            nth_link -= 1;
                        } else {
                            return Some(link_idx);
                        }
                    }
                }
            }
        }

        return None;
    }

/// # Complexity
///
/// O(1)

    pub fn random_link_index<R: Rng>(&self, rng: &mut R) -> Option<LinkIndex> {
        let n = self.link_count;
        if n > 0 {
            let link_idx: usize = rng.gen_range(0, n);
            return Some(LinkIndex(link_idx));
        }
        return None;
    }

    /// Iterate all outgoing links of `node_idx` and update the target nodes
    /// link count, as well as the global link count.
    ///
    /// # Complexity
    ///
    /// O(k), where `k` is the number of edges of `node_idx`.

    pub fn remove_all_outgoing_links_of_node(&mut self, node_idx: NodeIndex) {
        let mut links = Vec::with_capacity(self.node(node_idx).out_degree as usize);
        for (link_idx, _) in self.link_iter_for_node(node_idx) {
            links.push(link_idx);
        }
        for link in links {
            self.remove_link_at(link);
        }
    }

    /// Remove the node with index `node_idx` including
    /// all incoming and outgoing links.
    ///
    /// Moves the last node in the nodes array into the empty
    /// place and rewires all links. As such, this is a quite
    /// heavy operation!
    ///
    pub fn remove_node(&mut self, node_idx: NodeIndex) {
        //&self.nodes[node_idx.index()]
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
            links: List::empty(),
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

/// Returns a random link between two unconnected nodes, which would not
/// introduce
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

/// Returns true if the introduction of this directed link would lead towards a
/// cycle.
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

    pub fn disable_link_index(&mut self, link_idx: LinkIndex) -> bool {
        if self.link(link_idx).is_active() {
            self.active_link_count -= 1;
            self.link_mut(link_idx).active = false;
            return true;
        } else {
            return false;
        }
    }

    pub fn enable_link_index(&mut self, link_idx: LinkIndex) -> bool {
        if !self.link(link_idx).is_active() {
            self.active_link_count += 1;
            self.link_mut(link_idx).active = true;
            return true;
        } else {
            return false;
        }
    }

    pub fn disable_link(&mut self, source_node_idx: NodeIndex, target_node_idx: NodeIndex) -> bool {
        if let Some(link_idx) = self.find_link_index_exact(source_node_idx, target_node_idx) {
            return self.disable_link_index(link_idx);
        }
        return false;
    }

    pub fn enable_link(&mut self, source_node_idx: NodeIndex, target_node_idx: NodeIndex) -> bool {
        if let Some(link_idx) = self.find_link_index_exact(source_node_idx, target_node_idx) {
            return self.enable_link_index(link_idx);
        }
        return false;
    }

    pub fn first_link_of_node(&self, node_idx: NodeIndex) -> Option<&Link<L, EXTID>> {
        self.node(node_idx).links.head.map(|link_idx| self.link(link_idx))
    }

    pub fn last_link_of_node(&self, node_idx: NodeIndex) -> Option<&Link<L, EXTID>> {
        self.node(node_idx).links.tail.map(|link_idx| self.link(link_idx))
    }

    fn append(&mut self, node_idx: NodeIndex, link: Link<L, EXTID>) -> LinkIndex {
        match (self.node(node_idx).links.head, self.node(node_idx).links.tail) {
            (None, None) => {
                // append onto empty list
                let new_link_idx = self.allocate_link_item(LinkItem{
                    link: link,
                    prev: None,
                    next: None,
                });
                self.node_mut(node_idx).links = List {
                    head: Some(new_link_idx),
                    tail: Some(new_link_idx),
                };
                return new_link_idx;
            }
            (Some(_), Some(tail)) => {
                let new_link_idx = self.allocate_link_item(LinkItem {
                    link: link,
                    prev: Some(tail),
                    next: None,
                });
                assert!(self.links[tail.index()].next == None);
                self.links[tail.index()].next = Some(new_link_idx);
                self.node_mut(node_idx).links.tail = Some(new_link_idx);
                return new_link_idx;
            }
            _ => {
                panic!()
            }
        }
    }

    fn prepend(&mut self, node_idx: NodeIndex, link: Link<L, EXTID>) -> LinkIndex {
        match (self.node(node_idx).links.head, self.node(node_idx).links.tail) {
            (None, None) => {
                // prepend to empty list. same as append
                let new_link_idx = self.allocate_link_item(LinkItem{
                    link: link,
                    prev: None,
                    next: None,
                });
                self.node_mut(node_idx).links = List {
                    head: Some(new_link_idx),
                    tail: Some(new_link_idx),
                };
                return new_link_idx;
            }
            (Some(head), Some(_tail)) => {
                let new_link_idx = self.allocate_link_item(LinkItem {
                    link: link,
                    prev: None,
                    next: Some(head),
                });
                assert!(self.links[head.index()].prev == None);
                self.links[head.index()].prev = Some(new_link_idx);
                self.node_mut(node_idx).links.head = Some(new_link_idx);
                return new_link_idx;
            }
            _ => {
                panic!()
            }
        }
    }

    pub fn add_link(&mut self,
                    source_node_idx: NodeIndex,
                    target_node_idx: NodeIndex,
                    weight: L,
                    external_link_id: EXTID)
        -> LinkIndex {
            self.add_link_with_active(source_node_idx, target_node_idx, weight, external_link_id, true)
        }


    // This will destroy the ordering relation of links. Do not mix with `add_link` or `add_link_with_active`.

    pub fn add_link_unordered(&mut self,
                    source_node_idx: NodeIndex,
                    target_node_idx: NodeIndex,
                    weight: L,
                    external_link_id: EXTID)
        -> LinkIndex {
            if let Err(err) = self.valid_link(source_node_idx, target_node_idx) {
                panic!(err);
            }

            self.link_count += 1;
            self.active_link_count += 1;

            self.node_mut(source_node_idx).out_degree += 1;
            self.node_mut(target_node_idx).in_degree += 1;

            let link = Link {
                source_node_idx: source_node_idx,
                target_node_idx: target_node_idx,
                external_link_id: external_link_id,
                weight: weight,
                active: true,
            };

            return self.append(source_node_idx, link);
        }

    // Note: Doesn't check for cycles (except in the simple reflexive case).
// Note that we keep the list of links sorted according to it's
// external_link_id.
// XXX: Need test cases.
    pub fn add_link_with_active(&mut self,
                    source_node_idx: NodeIndex,
                    target_node_idx: NodeIndex,
                    weight: L,
                    external_link_id: EXTID,
                    active: bool)
        -> LinkIndex {
            if let Err(err) = self.valid_link(source_node_idx, target_node_idx) {
                panic!(err);
            }

            self.link_count += 1;
            if active {
                self.active_link_count += 1;
            }
            self.node_mut(source_node_idx).out_degree += 1;
            self.node_mut(target_node_idx).in_degree += 1;

            let link = Link {
                source_node_idx: source_node_idx,
                target_node_idx: target_node_idx,
                external_link_id: external_link_id,
                weight: weight,
                active: active,
            };

            match self.find_link_index_insert_before(source_node_idx, target_node_idx, external_link_id) {
                None => {
                    if let Some(tail) = self.node(source_node_idx).links.tail {
// check if last element is equal
                        if self.link(tail).target_node_idx == target_node_idx {
                            assert!(self.link(tail).external_link_id == external_link_id);
                            panic!("Duplicate link");
                        }
                    }
// append at end.
                    return self.append(source_node_idx, link);
                }
                Some(idx) => {
                    match self.link_item(idx).prev {
                        None => {
                            return self.prepend(source_node_idx, link);
                        }
                        Some(insert_after) => {
// check if previous element is not equal
                            if self.link(insert_after).target_node_idx == target_node_idx {
                                assert!(self.link(insert_after).external_link_id == external_link_id);
                                panic!("Duplicate link");
                            }

                            let new_link_idx = self.allocate_link_item(LinkItem{
                                link: link,
                                prev: Some(insert_after),
                                next: Some(idx),
                            });

                            self.links[insert_after.index()].next = Some(new_link_idx);
                            self.links[idx.index()].prev = Some(new_link_idx);

                            return new_link_idx;
                        }
                    }



                }
            }
        }

// Returns the index of the first element whoose external link id >
// `external_link_id`.
    fn find_link_index_insert_before(&self,
                                     source_node_idx: NodeIndex,
                                     _target_node_idx: NodeIndex,
                                     external_link_id: EXTID)
        -> Option<LinkIndex> {
// the links are sorted according to their external link id.
            let mut link_iter = self.link_iter_for_node(source_node_idx);
            for (idx, link) in &mut link_iter {
                if link.external_link_id() > external_link_id {
                    return Some(idx);
                }
            }
            return None;
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

/// Remove the link at index `link_index`.
    pub fn remove_link_at(&mut self, link_index: LinkIndex) {
        let found_idx = link_index;
        let source_node_idx = self.link(found_idx).source_node_idx;
        let target_node_idx = self.link(found_idx).target_node_idx;

        if self.link(found_idx).is_active() {
            self.active_link_count -= 1;
        }

// remove item from chain

        match (self.link_item(found_idx).prev, self.link_item(found_idx).next) {
            (None, None) => {
// Item is the only element of the list.
                assert!(self.node(source_node_idx).links.head == Some(found_idx));
                assert!(self.node(source_node_idx).links.tail == Some(found_idx));
                self.node_mut(source_node_idx).links = List::empty();
            }

            (None, Some(next)) => {
// Item is the first element in the list, followed by some other element.
                assert!(self.links[next.index()].prev == Some(found_idx));
                assert!(self.node(source_node_idx).links.head == Some(found_idx));
                assert!(self.node(source_node_idx).links.tail != Some(found_idx));
                self.node_mut(source_node_idx).links.head = Some(next);
                self.links[next.index()].prev = None;
            }

            (Some(prev), None) => {
// Item is the last element of the list, preceded by some other element.
                assert!(self.links[prev.index()].next == Some(found_idx));
                assert!(self.node(source_node_idx).links.tail == Some(found_idx));
                assert!(self.node(source_node_idx).links.head != Some(found_idx));

// make the previous element the new tail
                self.node_mut(source_node_idx).links.tail = Some(prev);
                self.links[prev.index()].next = None;
            }

            (Some(prev), Some(next)) => {
// Item is somewhere in the middle of the list. We don't have to
// update the head or tail pointers.
                assert!(self.node(source_node_idx).links.head != Some(found_idx));
                assert!(self.node(source_node_idx).links.tail != Some(found_idx));

                assert!(self.links[prev.index()].next == Some(found_idx));
                assert!(self.links[next.index()].prev == Some(found_idx));

                self.links[prev.index()].next = Some(next);
                self.links[next.index()].prev = Some(prev);
            }
        }

        self.links[found_idx.index()].next = None;
        self.links[found_idx.index()].prev = None;

// swap the item with the last one.
        let old_idx = LinkIndex(self.links.len() - 1);

        if found_idx == old_idx {
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
            let new_source_node_idx = self.link(new_idx).source_node_idx;

            match (self.link_item(new_idx).prev, self.link_item(new_idx).next) {
                (None, None) => {
// the moved element was the only element in the list.
                    assert!(self.node(new_source_node_idx).links.head == Some(old_idx));
                    assert!(self.node(new_source_node_idx).links.tail == Some(old_idx));

// Update both head and tail to the new element index.
                    self.node_mut(new_source_node_idx).links = List {
                        head: Some(new_idx),
                        tail: Some(new_idx)
                    };
                }

                (None, Some(next)) => {
// Item is the first element in the list, followed by some other element.
                    assert!(self.links[next.index()].prev == Some(old_idx));
                    assert!(self.node(new_source_node_idx).links.head == Some(old_idx));
                    assert!(self.node(new_source_node_idx).links.tail != Some(old_idx));
                    self.node_mut(new_source_node_idx).links.head = Some(new_idx);
                    self.links[next.index()].prev = Some(new_idx);
                }

                (Some(prev), None) => {
// Item is the last element of the list, preceded by some other element.
                    assert!(self.links[prev.index()].next == Some(old_idx));
                    assert!(self.node(new_source_node_idx).links.tail == Some(old_idx));
                    assert!(self.node(new_source_node_idx).links.head != Some(old_idx));

// make the previous element the new tail
                    self.node_mut(new_source_node_idx).links.tail = Some(new_idx);
                    self.links[prev.index()].next = Some(new_idx);
                }

                (Some(prev), Some(next)) => {
// Item is somewhere in the middle of the list. We don't have to
// update the head or tail pointers.
                    assert!(self.node(new_source_node_idx).links.head != Some(old_idx));
                    assert!(self.node(new_source_node_idx).links.tail != Some(old_idx));

                    assert!(self.links[prev.index()].next == Some(old_idx));
                    assert!(self.links[next.index()].prev == Some(old_idx));

                    self.links[prev.index()].next = Some(new_idx);
                    self.links[next.index()].prev = Some(new_idx);
                }
            }
        }

        assert!(self.node(source_node_idx).out_degree > 0);
        assert!(self.node(target_node_idx).in_degree > 0);
        self.node_mut(source_node_idx).out_degree -= 1;
        self.node_mut(target_node_idx).in_degree -= 1;
        self.link_count -= 1;
    }

/// Remove the first link that matches `source_node_idx` and `target_node_idx`.
/// XXX
    pub fn remove_link(&mut self, source_node_idx: NodeIndex, target_node_idx: NodeIndex) -> bool {
        if let Some(found_idx) = self.find_link_index_exact(source_node_idx, target_node_idx) {
            debug_assert!(self.link(found_idx).source_node_idx == source_node_idx);
            debug_assert!(self.link(found_idx).target_node_idx == target_node_idx);
            self.remove_link_at(found_idx);
            return true;
        }
        else {
// link was not found
            return false;
        }
    }
}

#[cfg(test)]
mod tests {
    use rand;
    use super::{ExternalId, Network, NodeType};

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

        assert_eq!(ExternalId(1),
                   g.first_link_of_node(i1).unwrap().external_link_id());
        assert_eq!(ExternalId(2),
                   g.last_link_of_node(i1).unwrap().external_link_id());

        assert_eq!(2, g.node(i1).out_degree());
        assert_eq!(1, g.node(h1).in_degree());
        assert_eq!(1, g.node(h2).in_degree());
        assert_eq!(2, g.link_count());

        assert_eq!(true, g.remove_link(i1, h1));
        assert_eq!(1, g.node(i1).out_degree());
        assert_eq!(0, g.node(h1).in_degree());
        assert_eq!(1, g.node(h2).in_degree());
        assert_eq!(1, g.link_count());

        assert_eq!(ExternalId(1),
                   g.first_link_of_node(i1).unwrap().external_link_id());
        assert_eq!(ExternalId(1),
                   g.last_link_of_node(i1).unwrap().external_link_id());

        assert_eq!(false, g.remove_link(i1, h1));
        assert_eq!(1, g.node(i1).out_degree());
        assert_eq!(0, g.node(h1).in_degree());
        assert_eq!(1, g.node(h2).in_degree());
        assert_eq!(1, g.link_count());

        assert_eq!(ExternalId(1),
                   g.first_link_of_node(i1).unwrap().external_link_id());
        assert_eq!(ExternalId(1),
                   g.last_link_of_node(i1).unwrap().external_link_id());

        assert_eq!(true, g.remove_link(i1, h2));
        assert_eq!(0, g.node(i1).out_degree());
        assert_eq!(0, g.node(h1).in_degree());
        assert_eq!(0, g.node(h2).in_degree());
        assert_eq!(0, g.link_count());

        assert!(g.first_link_of_node(i1).is_none());
        assert!(g.last_link_of_node(i1).is_none());

        // XXX: test for sort order
    }

    #[test]
    fn test_add_remove_link_unordered() {
        let mut g = Network::new();
        let i1 = g.add_node(NodeT::Input, ());
        let h1 = g.add_node(NodeT::Hidden, ());
        let h2 = g.add_node(NodeT::Hidden, ());

        g.add_link_unordered(i1, h1, 0.0, ());
        g.add_link_unordered(i1, h2, 0.0, ());

        assert_eq!(h1,
                   g.first_link_of_node(i1).unwrap().target_node_idx);
        assert_eq!(h2,
                   g.last_link_of_node(i1).unwrap().target_node_idx);

        assert_eq!(2, g.node(i1).out_degree());
        assert_eq!(1, g.node(h1).in_degree());
        assert_eq!(1, g.node(h2).in_degree());
        assert_eq!(2, g.link_count());

        assert_eq!(true, g.remove_link(i1, h1));
        assert_eq!(1, g.node(i1).out_degree());
        assert_eq!(0, g.node(h1).in_degree());
        assert_eq!(1, g.node(h2).in_degree());
        assert_eq!(1, g.link_count());

        assert_eq!(h2,
                   g.first_link_of_node(i1).unwrap().target_node_idx);
        assert_eq!(h2,
                   g.last_link_of_node(i1).unwrap().target_node_idx);

        assert_eq!(false, g.remove_link(i1, h1));
        assert_eq!(1, g.node(i1).out_degree());
        assert_eq!(0, g.node(h1).in_degree());
        assert_eq!(1, g.node(h2).in_degree());
        assert_eq!(1, g.link_count());

        assert_eq!(h2,
                   g.first_link_of_node(i1).unwrap().target_node_idx);
        assert_eq!(h2,
                   g.last_link_of_node(i1).unwrap().target_node_idx);

        assert_eq!(true, g.remove_link(i1, h2));
        assert_eq!(0, g.node(i1).out_degree());
        assert_eq!(0, g.node(h1).in_degree());
        assert_eq!(0, g.node(h2).in_degree());
        assert_eq!(0, g.link_count());

        assert!(g.first_link_of_node(i1).is_none());
        assert!(g.last_link_of_node(i1).is_none());
    }

    #[test]
    fn test_remove_all_outgoing_links() {
        let mut g = Network::new();
        let i1 = g.add_node(NodeT::Input, ());
        let h1 = g.add_node(NodeT::Hidden, ());
        let h2 = g.add_node(NodeT::Hidden, ());
        let o1 = g.add_node(NodeT::Output, ());

        g.add_link_unordered(i1, h1, 0.0, ());
        g.add_link_unordered(i1, h2, 0.0, ());
        g.add_link_unordered(h1, o1, 0.0, ());
        g.add_link_unordered(h2, o1, 0.0, ());

        assert_eq!(2, g.node(i1).out_degree());
        assert_eq!(1, g.node(h1).in_degree());
        assert_eq!(1, g.node(h2).in_degree());
        assert_eq!(1, g.node(h1).out_degree());
        assert_eq!(1, g.node(h2).out_degree());
        assert_eq!(2, g.node(o1).in_degree());
        assert_eq!(0, g.node(o1).out_degree());
        assert_eq!(4, g.link_count());

        g.remove_all_outgoing_links_of_node(i1);

        assert!(g.first_link_of_node(i1).is_none());
        assert!(g.last_link_of_node(i1).is_none());

        assert_eq!(0, g.node(i1).out_degree());
        assert_eq!(0, g.node(h1).in_degree());
        assert_eq!(0, g.node(h2).in_degree());
        assert_eq!(1, g.node(h1).out_degree());
        assert_eq!(1, g.node(h2).out_degree());
        assert_eq!(2, g.node(o1).in_degree());
        assert_eq!(0, g.node(o1).out_degree());
        assert_eq!(2, g.link_count());
    }


}
