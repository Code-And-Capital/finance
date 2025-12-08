from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Union
from copy import deepcopy


class Node:
    """
    Base class representing a node in a hierarchical strategy tree.

    Each node may contain children that represent either sub-strategies or
    securities. Nodes form a tree where every node maintains a reference to
    its parent and the root of the tree.

    Parameters
    ----------
    name : str
        The node's name.
    parent : Node, optional
        Parent node. If None, the node becomes the root of the tree.
    children : list, dict, optional
        Child nodes or string identifiers. A dict maps names to children.

    Attributes
    ----------
    name : str
        Node name.
    parent : Node
        Parent node reference.
    root : Node
        Top-level node of the tree.
    children : dict of str -> Node
        Actual instantiated child nodes.
    now : datetime or int
        Current timestamp used during backtests.
    stale : bool
        Flag indicating whether an update is required.
    full_name : str
        Node name including parent scopes.
    members : list of Node
        List of this node and all descendants.
    """

    def __init__(
        self,
        name: str,
        parent: Optional[Node] = None,
        children: Optional[Union[Dict[str, Any], Iterable[Any]]] = None,
    ) -> None:
        self.name = name
        self.children = {}
        self._lazy_children = {}
        self._universe_tickers = []
        self._childrenv = []
        self._original_children_are_present = (
            children is not None and len(children) >= 1
        )

        # strategy-child bookkeeping
        self._has_strat_children = False
        self._strat_children = []

        # root or child?
        if parent is None:
            self.parent = self
            self.root = self
            self.integer_positions = True
        else:
            self.parent = parent
            parent._add_children([self], dc=False)

        # add children if provided
        self._add_children(children, dc=True)

        # state
        self.now = 0
        self.root.stale = False

        # helper values
        self._price = 0
        self._value = 0
        self._notl_value = 0
        self._weight = 0
        self._capital = 0

        self._issec = False  # security flag
        self._fixed_income = False  # notional-weighting flag
        self._bidoffer_set = False
        self._bidoffer_paid = 0

    def __getitem__(self, key: str) -> Node:
        return self.children[key]

    def _add_children(
        self, children: Optional[Union[Dict[str, Any], Iterable[Any]]], dc: bool
    ) -> None:
        """
        Add new children to this node.

        Parameters
        ----------
        children : iterable, dict, or None
            Child nodes or ticker names. A dict preserves explicit names.
        dc : bool
            If True, deep-copies child nodes before attaching.
        """

        # Import here to avoid circular references
        from .security import Security
        from .strategy import StrategyBase

        if children is None:
            return

        # normalize dictionary mapping -> list
        if isinstance(children, dict):
            tmp = []
            for child_name, child_obj in children.items():
                if isinstance(child_obj, str):
                    tmp.append(child_name)
                else:
                    if dc:
                        child_obj = deepcopy(child_obj)
                    child_obj.name = child_name
                    tmp.append(child_obj)
            children = tmp

        # add each child
        for child in children:
            if dc:
                child = deepcopy(child)

            # string = lazy security definition
            if isinstance(child, str):
                if child in self._universe_tickers or child in self._lazy_children:
                    raise ValueError(f"Child {child} already exists")
                child = Security(child, lazy_add=True)

            # lazy-add object
            if getattr(child, "lazy_add", False):
                self._lazy_children[child.name] = child
                continue

            # immediate child creation
            if child.name in self.children:
                raise ValueError(f"Child {child.name} already exists")

            child.parent = self
            child._set_root(self.root)
            child.use_integer_positions(self.integer_positions)

            self.children[child.name] = child
            self._childrenv.append(child)

            # track strategy children
            if isinstance(child, StrategyBase):
                self._has_strat_children = True
                self._strat_children.append(child.name)
            elif child.name not in self._universe_tickers:
                self._universe_tickers.append(child.name)

    def _set_root(self, root: Node) -> None:
        """Propagate the root reference down the subtree."""
        self.root = root
        for child in self._childrenv:
            child._set_root(root)

    def use_integer_positions(self, integer_positions: bool) -> None:
        """
        Enable or disable integer-only positions.

        Parameters
        ----------
        integer_positions : bool
            Whether positions must be integers.
        """
        self.integer_positions = integer_positions
        for child in self._childrenv:
            child.use_integer_positions(integer_positions)

    @property
    def fixed_income(self) -> bool:
        """Return whether node uses notional weighting."""
        return self._fixed_income

    @property
    def prices(self):
        """Return the full price series for this node."""
        raise NotImplementedError()

    @property
    def price(self):
        """Return the latest price of the node."""
        raise NotImplementedError()

    @property
    def value(self) -> float:
        """Return current node value, updating if stale."""
        if self.root.stale:
            self.root.update(self.root.now, None)
        return self._value

    @property
    def notional_value(self) -> float:
        """Return current notional value, updating if stale."""
        if self.root.stale:
            self.root.update(self.root.now, None)
        return self._notl_value

    @property
    def weight(self) -> float:
        """Return weight relative to parent."""
        if self.root.stale:
            self.root.update(self.root.now, None)
        return self._weight

    def setup(self, universe: Any, **kwargs) -> None:
        """Initialize node from a universe."""
        raise NotImplementedError()

    def update(
        self, date: Any, data: Optional[Any] = None, inow: Optional[int] = None
    ) -> None:
        """Update node state for the given date."""
        raise NotImplementedError()

    def adjust(self, amount: float, update: bool = True, flow: bool = True) -> None:
        """Adjust node value by the given amount."""
        raise NotImplementedError()

    def allocate(self, amount: float, update: bool = True) -> None:
        """Allocate capital to this node."""
        raise NotImplementedError()

    @property
    def members(self) -> List["Node"]:
        """Return list of this node and all descendant nodes."""
        result = [self]
        for child in self.children.values():
            result.extend(child.members)
        return result

    @property
    def full_name(self) -> str:
        """Return hierarchical name including all parents."""
        if self.parent is self:
            return self.name
        return f"{self.parent.full_name}>{self.name}"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.full_name}>"

    def to_dot(self, root: bool = True) -> str:
        """
        Return a DOT language representation of this node and children.

        Parameters
        ----------
        root : bool
            Whether to wrap the output in a top-level `digraph` block.
        """
        name = lambda n: n.name or repr(n)
        edges = "\n".join(
            f'\t"{name(self)}" -> "{name(child)}"' for child in self.children.values()
        )
        below = "\n".join(child.to_dot(False) for child in self.children.values())
        body = "\n".join([edges, below]).rstrip()

        if root:
            return f"digraph {{\n{body}\n}}"
        return body
