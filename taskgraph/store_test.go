package taskgraph

import (
	"context"
	"sync"
)

// memStore is the tests' Store: the plan under a mutex, with a save that can
// fail. The package ships none; a host provides its own.
type memStore struct {
	mu   sync.Mutex
	g    Graph
	save func(g Graph) error
	// calls counts how many times Update ran fn.
	calls int
	// twice runs fn on a fresh copy a second time, as a store that retries a
	// conflicting transaction does, and keeps the second result.
	twice bool
}

func newMemStore(g Graph, save func(Graph) error) *memStore { return &memStore{g: g, save: save} }

func (s *memStore) View(context.Context) (Graph, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.g.Clone(), nil
}

func (s *memStore) Update(_ context.Context, fn func(g *Graph) error) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	next := s.g.Clone()
	s.calls++
	if err := fn(&next); err != nil {
		return err
	}
	if s.twice {
		next = s.g.Clone()
		s.calls++
		if err := fn(&next); err != nil {
			return err
		}
	}
	if s.save != nil {
		if err := s.save(next); err != nil {
			return err
		}
	}
	s.g = next
	return nil
}
