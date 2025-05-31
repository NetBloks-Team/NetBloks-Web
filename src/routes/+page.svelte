<script lang="ts">
    let count = $state(0);
</script>

<p>You have clicked {count} times</p>
<button
    onclick={() => {
        count++;
        console.log(count);
    }}>Click me!</button
>
