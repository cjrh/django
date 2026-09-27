import asyncio
import threading

from asgiref.sync import sync_to_async

from django.db import connection, connections, transaction
from django.test import TestCase, TransactionTestCase, skipUnlessDBFeature

from .models import Reporter


def current_thread_and_connection():
    return threading.current_thread(), connections["default"]


@skipUnlessDBFeature("uses_savepoints")
class AsyncAtomicTests(TransactionTestCase):
    available_apps = ["transactions"]

    async def test_commit(self):
        async with transaction.atomic():
            await Reporter.objects.acreate(first_name="Tintin")
        self.assertEqual(
            [r.first_name async for r in Reporter.objects.all()], ["Tintin"]
        )

    async def test_rollback(self):
        with self.assertRaisesMessage(Exception, "Oops"):
            async with transaction.atomic():
                await Reporter.objects.acreate(first_name="Haddock")
                raise Exception("Oops, that's his last name")
        self.assertEqual(await Reporter.objects.acount(), 0)

    async def test_nested_rollback_commit(self):
        async with transaction.atomic():
            await Reporter.objects.acreate(last_name="Tintin")
            with self.assertRaisesMessage(Exception, "Oops"):
                async with transaction.atomic():
                    await Reporter.objects.acreate(first_name="Haddock")
                    raise Exception("Oops, that's his last name")
        self.assertEqual(
            [r.last_name async for r in Reporter.objects.all()], ["Tintin"]
        )

    async def test_nested_commit_rollback(self):
        with self.assertRaisesMessage(Exception, "Oops"):
            async with transaction.atomic():
                async with transaction.atomic():
                    await Reporter.objects.acreate(first_name="Tintin")
                raise Exception("Oops, that's his first name")
        self.assertEqual(await Reporter.objects.acount(), 0)

    async def test_reuse_nested(self):
        atomic = transaction.atomic()
        async with atomic:
            await Reporter.objects.acreate(first_name="Tintin")
            with self.assertRaisesMessage(Exception, "Oops"):
                async with atomic:
                    await Reporter.objects.acreate(first_name="Haddock")
                    raise Exception("Oops, that's his last name")
        self.assertEqual(await Reporter.objects.acount(), 1)

    async def test_sync_atomic_inside_async_atomic(self):
        def create_and_fail():
            with transaction.atomic():
                Reporter.objects.create(first_name="Haddock")
                raise Exception("Oops, that's his last name")

        async with transaction.atomic():
            await Reporter.objects.acreate(first_name="Tintin")
            with self.assertRaisesMessage(Exception, "Oops"):
                await sync_to_async(create_and_fail)()
        self.assertEqual(await Reporter.objects.acount(), 1)

    async def test_outermost_block_uses_own_thread_and_connection(self):
        get = sync_to_async(current_thread_and_connection)
        outer_thread, outer_connection = await get()
        async with transaction.atomic():
            block_thread, block_connection = await get()
            async with transaction.atomic():
                nested_thread, nested_connection = await get()
            # Leaving a nested block keeps the outermost block's worker.
            self.assertEqual(await get(), (block_thread, block_connection))
        self.assertEqual(await get(), (outer_thread, outer_connection))
        self.assertIsNot(block_thread, outer_thread)
        self.assertIsNot(block_connection, outer_connection)
        self.assertIs(nested_thread, block_thread)
        self.assertIs(nested_connection, block_connection)

    async def test_error_on_enter(self):
        async with transaction.atomic():
            await Reporter.objects.acreate(first_name="Tintin")
            with self.assertRaisesMessage(RuntimeError, "durable"):
                async with transaction.atomic(durable=True):
                    pass
            # The enclosing block is still usable.
            await Reporter.objects.acreate(first_name="Haddock")
        self.assertEqual(await Reporter.objects.acount(), 2)

    async def test_on_commit(self):
        callbacks = []
        async with transaction.atomic():
            await sync_to_async(transaction.on_commit)(lambda: callbacks.append(1))
            self.assertEqual(callbacks, [])
        self.assertEqual(callbacks, [1])

    async def test_outer_connection_not_in_transaction(self):
        async with transaction.atomic():
            self.assertIs(
                await sync_to_async(lambda: connection.in_atomic_block)(), True
            )
        self.assertIs(await sync_to_async(lambda: connection.in_atomic_block)(), False)


# The database must allow a transaction on one connection while other
# connections write, so these tests can't use an in-memory SQLite database.
@skipUnlessDBFeature("uses_savepoints", "test_db_allows_multiple_connections")
class AsyncAtomicIsolationTests(TransactionTestCase):
    available_apps = ["transactions"]

    async def test_connection_closed_on_exit(self):
        async with transaction.atomic():
            await Reporter.objects.acreate(first_name="Tintin")
            _, block_connection = await sync_to_async(current_thread_and_connection)()
            self.assertIsNotNone(block_connection.connection)
        self.assertIsNone(block_connection.connection)

    async def test_rollback_does_not_undo_other_task(self):
        transaction_started = asyncio.Event()
        other_write_finished = asyncio.Event()

        async def task_one():
            with self.assertRaisesMessage(ValueError, "Undo my work"):
                async with transaction.atomic():
                    await Reporter.objects.acreate(first_name="Tintin")
                    transaction_started.set()
                    await other_write_finished.wait()
                    raise ValueError("Undo my work")

        async def task_two():
            await transaction_started.wait()
            # This is outside the atomic block.
            await Reporter.objects.acreate(first_name="Haddock")
            other_write_finished.set()

        await asyncio.gather(task_one(), task_two())
        self.assertEqual(
            [r.first_name async for r in Reporter.objects.all()], ["Haddock"]
        )

    async def test_concurrent_transactions(self):
        both_started = asyncio.Barrier(2)

        async def create(name, fail):
            async with transaction.atomic():
                await Reporter.objects.acreate(first_name=name)
                await both_started.wait()
                if fail:
                    raise ValueError(name)

        results = await asyncio.gather(
            create("Tintin", fail=False),
            create("Haddock", fail=True),
            return_exceptions=True,
        )
        self.assertIsNone(results[0])
        self.assertIsInstance(results[1], ValueError)
        self.assertEqual(
            [r.first_name async for r in Reporter.objects.all()], ["Tintin"]
        )


@skipUnlessDBFeature("uses_savepoints")
class AsyncAtomicInsideTestCaseTests(TestCase):
    """
    TestCase runs async tests with async_to_sync(), inside an atomic block on
    the test thread. An async atomic block must join that transaction.
    """

    @classmethod
    def setUpTestData(cls):
        Reporter.objects.create(first_name="Tintin")

    async def test_sees_test_data(self):
        async with transaction.atomic():
            self.assertEqual(await Reporter.objects.acount(), 1)

    async def test_uses_test_connection(self):
        get = sync_to_async(current_thread_and_connection)
        test_thread_and_connection = await get()
        async with transaction.atomic():
            self.assertEqual(await get(), test_thread_and_connection)

    async def test_rollback(self):
        with self.assertRaisesMessage(Exception, "Oops"):
            async with transaction.atomic():
                await Reporter.objects.acreate(first_name="Haddock")
                raise Exception("Oops, that's his last name")
        self.assertEqual(await Reporter.objects.acount(), 1)

    async def test_durable(self):
        async with transaction.atomic(durable=True):
            await Reporter.objects.acreate(first_name="Haddock")
        self.assertEqual(await Reporter.objects.acount(), 2)
