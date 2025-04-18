module wanderer::domain_market {

    use sui::object::{Self, UID};
    use sui::tx_context::{Self, TxContext};
    use sui::transfer;
    use std::vector;

    /// A domain entry listed for sale
    public struct DomainEntry has key, store {
        owner: address,
        domain: vector<u8>, // domain as string bytes
        ip: vector<u8>,     // IP address as string bytes
        price: u64
    }

    /// The registry object holding all domains
    public struct DomainRegistry has key, store {
        id: UID,
        domains: vector<DomainEntry>
    }

    /// Create the central registry (admin function)
    public fun create_registry(ctx: &mut TxContext) {
        let registry = DomainRegistry {
            id: object::new(ctx),
            domains: vector::empty<DomainEntry>()
        };
        transfer::share_object(registry);
    }

    /// Register a domain for sale
    public fun register_domain(
        registry: &mut DomainRegistry,
        domain: vector<u8>,
        ip: vector<u8>,
        price: u64,
        ctx: &mut TxContext
    ) {
        let entry = DomainEntry {
            owner: ctx.sender(),
            domain,
            ip,
            price
        };
        vector::push_back(&mut registry.domains, entry);
    }

    /// Buy a domain by domain name (searching in list)
    public fun buy_domain(
        registry: &mut DomainRegistry,
        domain: vector<u8>,
        buyer: address
    ) {
        let len = vector::length(&registry.domains);
        let mut i = 0;
        while (i < len) {
            let entry = vector::borrow_mut(&mut registry.domains, i);
            if (entry.domain == domain) {
                // Transfer ownership
                entry.owner = buyer;
                // Optionally remove from list if it's a one-time sale
                // vector::remove(&mut registry.domains, i);
                break;
            };
            i = i + 1;
        };
    }

    /// Get all domains (read-only)
    public fun get_domains(registry: &DomainRegistry): vector<DomainEntry> {
        registry.domains
    }
}
