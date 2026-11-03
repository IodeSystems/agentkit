package taskgraph

import "context"

// Store is where a plan lives. The integration provides it; this package ships
// no implementation, the same as agent.Store. A file beside a session, a row in
// a database and a map under a mutex are all the host's choice.
//
// The contract, which this package cannot enforce:
//
//   - One Store per plan, shared by every session that works the plan. A main
//     agent and its sub-agents call the same Store; a copy per sub-agent loses
//     updates with no error, since each Update sees only its own copy.
//   - Update is atomic against every other Update and View of the same plan:
//     fn reads the plan as it is, and its change is kept whole or not at all.
//   - Update keeps the change only if fn returns nil and the store's own write
//     succeeds; either failure is Update's error and the plan is as it was.
//   - fn may be called more than once (a store that retries a conflicting
//     transaction), so fn must do nothing but change g. Apply relies on this.
//   - View is a copy: a caller may change what it returns.
type Store interface {
	View(ctx context.Context) (Graph, error)
	Update(ctx context.Context, fn func(g *Graph) error) error
}
