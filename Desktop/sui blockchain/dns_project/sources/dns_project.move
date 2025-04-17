module dns_project::dns {

    use sui::object::{Self, UID};
    use sui::tx_context::{Self, TxContext};
    use sui::transfer;

    struct Domain has key {
        id: UID,
        name: vector<u8>,
        ip: vector<u8>,
        owner: address,
    }

    public fun register_domain(
        name: vector<u8>, 
        ip: vector<u8>, 
        ctx: &mut TxContext
    ): Domain {
        let sender = tx_context::sender(ctx);
        let uid = object::new(ctx);
        Domain {
            id: uid,
            name,
            ip,
            owner: sender,
        }
    }
}
