"""
Blockchain-based audit logging module for federated learning.
"""

import json
import time
from typing import Dict, Any, Optional
from web3 import Web3
from eth_account import Account
from pathlib import Path
import os
from logging import getLogger

logger = getLogger(__name__)

# Smart contract ABI for audit logging
AUDIT_CONTRACT_ABI = [
    {
        "inputs": [],
        "stateMutability": "nonpayable",
        "type": "constructor"
    },
    {
        "anonymous": False,
        "inputs": [
            {
                "indexed": True,
                "internalType": "address",
                "name": "client",
                "type": "address"
            },
            {
                "indexed": False,
                "internalType": "string",
                "name": "eventType",
                "type": "string"
            },
            {
                "indexed": False,
                "internalType": "string",
                "name": "data",
                "type": "string"
            }
        ],
        "name": "AuditEvent",
        "type": "event"
    },
    {
        "inputs": [
            {
                "internalType": "string",
                "name": "eventType",
                "type": "string"
            },
            {
                "internalType": "string",
                "name": "data",
                "type": "string"
            }
        ],
        "name": "logEvent",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]

class BlockchainAuditor:
    """Blockchain-based audit logging system."""
    
    def __init__(
        self,
        provider_url: str = "http://localhost:8545",
        contract_address: Optional[str] = None,
        private_key: Optional[str] = None
    ):
        """Initialize the blockchain auditor.
        
        Args:
            provider_url: URL of the Ethereum provider (e.g., Ganache)
            contract_address: Address of deployed audit contract
            private_key: Private key for transaction signing
        """
        self.mock_mode = False
        
        try:
            self.w3 = Web3(Web3.HTTPProvider(provider_url))
            
            # Check connection
            if not self.w3.is_connected():
                logger.warning(f"Cannot connect to blockchain at {provider_url}, using mock mode")
                self.mock_mode = True
                return
            
            # Load or create account
            if private_key:
                self.account = Account.from_key(private_key)
            else:
                self.account = Account.create()
            
            # Deploy contract if address not provided
            if contract_address:
                self.contract_address = contract_address
                self.contract = self.w3.eth.contract(
                    address=contract_address,
                    abi=AUDIT_CONTRACT_ABI
                )
            else:
                self.contract_address = self._deploy_contract()
                
        except Exception as e:
            logger.warning(f"Blockchain initialization failed: {e}, using mock mode")
            self.mock_mode = True
    
    def _deploy_contract(self) -> str:
        """Deploy the audit logging contract.
        
        Returns:
            Address of the deployed contract
        """
        if self.mock_mode:
            mock_address = "0x1234567890123456789012345678901234567890"
            logger.info(f"Mock contract deployed at: {mock_address}")
            return mock_address
            
        # Get bytecode - if it's just a placeholder, use mock mode
        bytecode = self._get_contract_bytecode()
        if bytecode == "0x...":
            logger.warning("No real bytecode available, switching to mock mode")
            self.mock_mode = True
            return self._deploy_contract()
            
        contract = self.w3.eth.contract(
            abi=AUDIT_CONTRACT_ABI,
            bytecode=bytecode
        )
        
        # Build transaction
        transaction = contract.constructor().build_transaction({
            'from': self.account.address,
            'nonce': self.w3.eth.get_transaction_count(self.account.address),
            'gas': 2000000,
            'gasPrice': self.w3.eth.gas_price
        })
        
        # Sign and send transaction
        signed_txn = self.w3.eth.account.sign_transaction(
            transaction,
            self.account.key
        )
        tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        
        # Wait for transaction receipt
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        return tx_receipt.contractAddress
    
    def _get_contract_bytecode(self) -> str:
        """Get the bytecode for the audit contract.
        
        Returns:
            Contract bytecode as string
        """
        # In a real implementation, this would load the compiled contract bytecode
        # For demonstration, using a placeholder
        return "0x..."  # Replace with actual bytecode
    
    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Log an event to the blockchain.
        
        Args:
            event_type: Type of event (e.g., 'round_complete', 'trust_update')
            data: Event data to log
        """
        if self.mock_mode:
            # Mock logging for demonstration
            import time
            mock_hash = f"0x{hash(f'{event_type}{json.dumps(data)}{time.time()}') % (2**64):016x}"
            logger.info(f"Mock event logged: {event_type} - Hash: {mock_hash}")
            return {"transactionHash": mock_hash, "blockNumber": int(time.time()) % 10000}
            
        try:
            # Convert data to JSON string
            data_str = json.dumps(data)
            
            # Build transaction
            transaction = self.contract.functions.logEvent(
                event_type,
                data_str
            ).build_transaction({
                'from': self.account.address,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
                'gas': 200000,
                'gasPrice': self.w3.eth.gas_price
            })
            
            # Sign and send transaction
            signed_txn = self.w3.eth.account.sign_transaction(
                transaction,
                self.account.key
            )
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for transaction receipt
            tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            logger.info(f"Event logged successfully: {tx_receipt.transactionHash.hex()}")
            return tx_receipt
            
        except Exception as e:
            logger.error(f"Error logging event: {str(e)}")
            raise
    
    def get_events(
        self,
        event_type: Optional[str] = None,
        from_block: int = 0
    ) -> list:
        """Retrieve events from the blockchain.
        
        Args:
            event_type: Filter by event type
            from_block: Starting block number
            
        Returns:
            List of matching events
        """
        if self.mock_mode:
            # Return mock events for demonstration
            import time
            current_time = int(time.time())
            mock_events = [
                {
                    "event_type": "system_init",
                    "data": {"status": "initialized", "version": "1.0"},
                    "timestamp": current_time - 3600,  # 1 hour ago
                    "block_number": 1000,
                    "transaction_hash": "0x1234567890abcdef"
                },
                {
                    "event_type": "round_complete",
                    "data": {"round_id": 1, "participants": 3, "accuracy": 0.89},
                    "timestamp": current_time - 1800,  # 30 minutes ago
                    "block_number": 1001,
                    "transaction_hash": "0xabcdef1234567890"
                },
                {
                    "event_type": "trust_update",
                    "data": {"client_id": "client_1", "trust_score": 0.95},
                    "timestamp": current_time - 900,  # 15 minutes ago
                    "block_number": 1002,
                    "transaction_hash": "0xfedcba0987654321"
                }
            ]
            if event_type:
                mock_events = [e for e in mock_events if e["event_type"] == event_type]
            return mock_events
            
        try:
            # Create event filter
            event_filter = self.contract.events.AuditEvent.create_filter(
                fromBlock=from_block
            )
            
            # Get all events
            events = event_filter.get_all_entries()
            
            # Filter by event type if specified
            if event_type:
                events = [
                    event for event in events
                    if event['args']['eventType'] == event_type
                ]
            
            # Parse event data
            parsed_events = []
            for event in events:
                parsed_events.append({
                    'client': event['args']['client'],
                    'event_type': event['args']['eventType'],
                    'data': json.loads(event['args']['data']),
                    'block_number': event['blockNumber'],
                    'transaction_hash': event['transactionHash'].hex(),
                    'timestamp': self.w3.eth.get_block(event['blockNumber'])['timestamp']
                })
            
            return parsed_events
            
        except Exception as e:
            logger.error(f"Error retrieving events: {str(e)}")
            raise
    
    def verify_event(self, transaction_hash: str) -> Dict[str, Any]:
        """Verify an event's authenticity.
        
        Args:
            transaction_hash: Hash of the transaction to verify
            
        Returns:
            Dictionary containing verification results
        """
        try:
            # Get transaction receipt
            receipt = self.w3.eth.get_transaction_receipt(transaction_hash)
            
            # Get transaction
            transaction = self.w3.eth.get_transaction(transaction_hash)
            
            # Get block information
            block = self.w3.eth.get_block(receipt['blockNumber'])
            
            # Decode event logs
            logs = self.contract.events.AuditEvent().process_receipt(receipt)
            
            return {
                'verified': True,
                'block_number': receipt['blockNumber'],
                'block_hash': block['hash'].hex(),
                'timestamp': block['timestamp'],
                'gas_used': receipt['gasUsed'],
                'status': receipt['status'],
                'event_data': [log['args'] for log in logs]
            }
            
        except Exception as e:
            logger.error(f"Error verifying event: {str(e)}")
            return {
                'verified': False,
                'error': str(e)
            }